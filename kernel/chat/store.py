"""Conversation Store — Redis primary, JSONL fallback.

Provides server-side conversation history so chats survive page
refreshes and Railway redeploys.

Backend priority:
  1. Redis (via REDIS_URL) — fast, shared, TTL-managed
  2. JSONL on /data volume — append-only files, works without Redis

Storage layout (JSONL fallback):
    {data_dir}/conversations/
        {conversation_id}.jsonl   — one JSON line per message
        _index.json               — conversation list with metadata

Redis key layout:
    {prefix}conv:{id}:meta   — JSON string with conversation metadata
    {prefix}conv:{id}:msgs   — list of JSON-encoded messages
    {prefix}convindex        — sorted set of conv IDs scored by updated_at
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

try:
    import redis  # type: ignore[import-untyped]
except ImportError:
    redis = None  # type: ignore[assignment]

from ..config.settings import settings

logger = logging.getLogger("vex.chat.store")

# Build Redis exception set once — used by make_conversation_store()
_REDIS_ERRORS: tuple[type[BaseException], ...] = (ConnectionError, OSError, RuntimeError)
if redis is not None:
    _REDIS_ERRORS = (*_REDIS_ERRORS, redis.exceptions.RedisError)

MAX_CONVERSATIONS = 200
MAX_MESSAGES_PER_CONVERSATION = 500

# Token estimation: 1 token ≈ 4 characters (GPT-class models)
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length. ~1 token per 4 chars."""
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class Message:
    """A single chat message."""

    id: str
    role: str  # 'user' | 'vex'
    content: str
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize message to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """Deserialize a message from a plain dict."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", ""),
            metadata=data.get("metadata", {}),
            token_count=data.get("token_count", 0),
        )

    def to_llm_message(self) -> dict[str, str]:
        """Convert to LLM-compatible message dict."""
        role = "assistant" if self.role == "vex" else self.role
        return {"role": role, "content": self.content}


@dataclass
class Conversation:
    """Conversation metadata for the index."""

    id: str
    title: str
    created_at: float
    updated_at: float
    message_count: int = 0
    preview: str = ""
    total_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize conversation metadata to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Conversation:
        """Deserialize conversation metadata from a plain dict."""
        return cls(
            id=data["id"],
            title=data.get("title", "Untitled"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            message_count=data.get("message_count", 0),
            preview=data.get("preview", ""),
            total_tokens=data.get("total_tokens", 0),
        )


# ═══════════════════════════════════════════════════════════════
#  REDIS BACKEND
# ═══════════════════════════════════════════════════════════════


class _RedisBackend:
    """Redis-backed conversation storage."""

    def __init__(self, url: str, prefix: str, ttl: int) -> None:
        import redis as _redis

        self._r = _redis.from_url(url, decode_responses=True)
        self._prefix = prefix
        self._ttl = ttl
        # Probe connectivity
        self._r.ping()

    def _meta_key(self, conv_id: str) -> str:
        return f"{self._prefix}conv:{conv_id}:meta"

    def _msgs_key(self, conv_id: str) -> str:
        return f"{self._prefix}conv:{conv_id}:msgs"

    def _index_key(self) -> str:
        return f"{self._prefix}convindex"

    def create_conversation(self, conv_id: str | None = None) -> str:
        cid = conv_id or str(uuid.uuid4())
        now = time.time()
        meta = Conversation(id=cid, title="New conversation", created_at=now, updated_at=now)
        pipe = self._r.pipeline()
        pipe.set(self._meta_key(cid), json.dumps(meta.to_dict()))
        pipe.expire(self._meta_key(cid), self._ttl)
        pipe.zadd(self._index_key(), {cid: now})
        pipe.execute()
        self._prune_old()
        return cid

    def append_message(self, conv_id: str, message: Message) -> None:
        if message.token_count == 0:
            message.token_count = estimate_tokens(message.content)

        # Fold meta fetch + message append into a single round-trip
        pipe = self._r.pipeline()
        pipe.get(self._meta_key(conv_id))
        pipe.rpush(self._msgs_key(conv_id), json.dumps(message.to_dict()))
        pipe.expire(self._msgs_key(conv_id), self._ttl)
        results = pipe.execute()

        raw = cast(str | None, results[0])
        if raw is None:
            # Lazily create conversation — avoids separate EXISTS call
            self.create_conversation(conv_id)
            raw = cast(str | None, self._r.get(self._meta_key(conv_id)))

        if raw:
            meta = Conversation.from_dict(json.loads(raw))
        else:
            now = time.time()
            meta = Conversation(
                id=conv_id, title="New conversation", created_at=now, updated_at=now
            )

        meta.updated_at = time.time()
        meta.message_count += 1
        meta.total_tokens += message.token_count
        meta.preview = message.content[:100]

        if message.role == "user" and meta.title == "New conversation":
            meta.title = message.content[:60]
            if len(message.content) > 60:
                meta.title += "..."

        pipe2 = self._r.pipeline()
        pipe2.set(self._meta_key(conv_id), json.dumps(meta.to_dict()))
        pipe2.expire(self._meta_key(conv_id), self._ttl)
        pipe2.zadd(self._index_key(), {conv_id: meta.updated_at})
        pipe2.execute()

    def get_conversation(self, conv_id: str) -> dict[str, Any] | None:
        raw = cast(str | None, self._r.get(self._meta_key(conv_id)))
        if not raw:
            return None
        meta = json.loads(raw)
        raw_msgs = cast(
            list[str], self._r.lrange(self._msgs_key(conv_id), -MAX_MESSAGES_PER_CONVERSATION, -1)
        )
        messages = []
        for m in raw_msgs:
            try:
                messages.append(Message.from_dict(json.loads(m)).to_dict())
            except (json.JSONDecodeError, KeyError):
                continue
        return {**meta, "messages": messages}

    def list_conversations(self, limit: int = 50) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        # Sorted set: highest score (most recent) first
        ids = cast(list[str], self._r.zrevrange(self._index_key(), 0, limit - 1))
        result: list[dict[str, Any]] = []
        if ids:
            pipe = self._r.pipeline()
            for cid in ids:
                pipe.get(self._meta_key(cid))
            raws = pipe.execute()
            for raw in raws:
                if raw:
                    try:
                        result.append(json.loads(raw))
                    except json.JSONDecodeError:
                        continue
        return result

    def delete_conversation(self, conv_id: str) -> bool:
        existed = self._r.exists(self._meta_key(conv_id))
        pipe = self._r.pipeline()
        pipe.delete(self._meta_key(conv_id))
        pipe.delete(self._msgs_key(conv_id))
        pipe.zrem(self._index_key(), conv_id)
        pipe.execute()
        return bool(existed)

    def get_llm_messages(self, conv_id: str, max_tokens: int = 28000) -> list[dict[str, str]]:
        raw_msgs = cast(list[str], self._r.lrange(self._msgs_key(conv_id), 0, -1))
        if not raw_msgs:
            return []
        messages: list[Message] = []
        for m in raw_msgs:
            try:
                messages.append(Message.from_dict(json.loads(m)))
            except (json.JSONDecodeError, KeyError):
                continue
        result: list[dict[str, str]] = []
        token_sum = 0
        for msg in reversed(messages):
            tc = msg.token_count or estimate_tokens(msg.content)
            if token_sum + tc > max_tokens:
                break
            result.append(msg.to_llm_message())
            token_sum += tc
        result.reverse()
        return result

    def get_token_count(self, conv_id: str) -> int:
        raw = cast(str | None, self._r.get(self._meta_key(conv_id)))
        if raw:
            meta = json.loads(raw)
            return int(meta.get("total_tokens", 0))
        return 0

    def _prune_old(self) -> None:
        count = cast(int, self._r.zcard(self._index_key()))
        if count <= MAX_CONVERSATIONS:
            return
        to_remove = count - MAX_CONVERSATIONS
        # Remove oldest (lowest scores)
        oldest = cast(list[str], self._r.zrange(self._index_key(), 0, to_remove - 1))
        if not oldest:
            return
        pipe = self._r.pipeline()
        for cid in oldest:
            pipe.delete(self._meta_key(cid))
            pipe.delete(self._msgs_key(cid))
        pipe.zrem(self._index_key(), *oldest)
        pipe.execute()
        logger.info("Pruned %d old conversations from Redis", to_remove)


# ═══════════════════════════════════════════════════════════════
#  JSONL BACKEND (fallback)
# ═══════════════════════════════════════════════════════════════


class _JSONLBackend:
    """JSONL-based conversation persistence on filesystem."""

    def __init__(self, data_dir: str | None = None) -> None:
        self._base_dir = Path(data_dir or settings.data_dir) / "conversations"
        try:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            probe = self._base_dir / ".write_probe"
            probe.touch()
            probe.unlink()
        except OSError:
            self._base_dir = Path("/tmp/vex-conversations")
            self._base_dir.mkdir(parents=True, exist_ok=True)
            logger.warning("Data volume not writable — using ephemeral %s", self._base_dir)
        self._index_path = self._base_dir / "_index.json"
        self._index: dict[str, Conversation] = {}
        self._load_index()

    def _load_index(self) -> None:
        if self._index_path.exists():
            try:
                with open(self._index_path, encoding="utf-8") as f:
                    data = json.load(f)
                for item in data.get("conversations", []):
                    conv = Conversation.from_dict(item)
                    self._index[conv.id] = conv
                logger.info("Loaded %d conversations from index", len(self._index))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Corrupt conversation index, rebuilding: %s", e)
                self._rebuild_index()
        else:
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._index = {}
        for path in self._base_dir.glob("*.jsonl"):
            conv_id = path.stem
            if conv_id.startswith("_"):
                continue
            messages = self._read_messages(conv_id)
            if messages:
                first_user = next((m for m in messages if m.role == "user"), None)
                title = (
                    (first_user.content[:60] + "...")
                    if first_user and len(first_user.content) > 60
                    else (first_user.content if first_user else "Untitled")
                )
                self._index[conv_id] = Conversation(
                    id=conv_id,
                    title=title,
                    created_at=(
                        float(messages[0].timestamp)
                        if messages[0].timestamp.replace(".", "").isdigit()
                        else time.time()
                    ),
                    updated_at=(
                        float(messages[-1].timestamp)
                        if messages[-1].timestamp.replace(".", "").isdigit()
                        else time.time()
                    ),
                    message_count=len(messages),
                    preview=messages[-1].content[:100],
                )
        self._save_index()
        logger.info("Rebuilt conversation index: %d conversations", len(self._index))

    def _save_index(self) -> None:
        data = {
            "conversations": [
                conv.to_dict()
                for conv in sorted(
                    self._index.values(),
                    key=lambda c: c.updated_at,
                    reverse=True,
                )
            ]
        }
        try:
            with open(self._index_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            logger.error("Failed to save conversation index: %s", e)

    def _conv_path(self, conv_id: str) -> Path:
        return self._base_dir / f"{conv_id}.jsonl"

    def _read_messages(self, conv_id: str) -> list[Message]:
        path = self._conv_path(conv_id)
        if not path.exists():
            return []
        messages: list[Message] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    messages.append(Message.from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError):
                    continue
        return messages

    def create_conversation(self, conv_id: str | None = None) -> str:
        cid = conv_id or str(uuid.uuid4())
        now = time.time()
        self._index[cid] = Conversation(
            id=cid, title="New conversation", created_at=now, updated_at=now
        )
        try:
            self._conv_path(cid).touch()
        except OSError as e:
            logger.warning("Could not create conversation file %s: %s", cid, e)
        self._save_index()
        self._prune_old()
        return cid

    def append_message(self, conv_id: str, message: Message) -> None:
        if conv_id not in self._index:
            self.create_conversation(conv_id)

        if message.token_count == 0:
            message.token_count = estimate_tokens(message.content)

        path = self._conv_path(conv_id)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(message.to_dict()) + "\n")
        except OSError as e:
            logger.error("Failed to write message to %s: %s", conv_id, e)
            return

        conv = self._index[conv_id]
        conv.updated_at = time.time()
        conv.message_count += 1
        conv.total_tokens += message.token_count
        conv.preview = message.content[:100]

        if message.role == "user" and conv.title == "New conversation":
            conv.title = message.content[:60]
            if len(message.content) > 60:
                conv.title += "..."

        self._save_index()

    def get_conversation(self, conv_id: str) -> dict[str, Any] | None:
        if conv_id not in self._index:
            return None
        conv = self._index[conv_id]
        messages = self._read_messages(conv_id)
        return {
            **conv.to_dict(),
            "messages": [m.to_dict() for m in messages[-MAX_MESSAGES_PER_CONVERSATION:]],
        }

    def list_conversations(self, limit: int = 50) -> list[dict[str, Any]]:
        sorted_convs = sorted(
            self._index.values(),
            key=lambda c: c.updated_at,
            reverse=True,
        )
        return [c.to_dict() for c in sorted_convs[:limit]]

    def delete_conversation(self, conv_id: str) -> bool:
        if conv_id not in self._index:
            return False
        del self._index[conv_id]
        path = self._conv_path(conv_id)
        if path.exists():
            path.unlink()
        self._save_index()
        return True

    def get_llm_messages(self, conv_id: str, max_tokens: int = 28000) -> list[dict[str, str]]:
        messages = self._read_messages(conv_id)
        if not messages:
            return []
        result: list[dict[str, str]] = []
        token_sum = 0
        for msg in reversed(messages):
            tc = msg.token_count or estimate_tokens(msg.content)
            if token_sum + tc > max_tokens:
                break
            result.append(msg.to_llm_message())
            token_sum += tc
        result.reverse()
        return result

    def get_token_count(self, conv_id: str) -> int:
        if conv_id in self._index:
            return self._index[conv_id].total_tokens
        return 0

    def _prune_old(self) -> None:
        if len(self._index) <= MAX_CONVERSATIONS:
            return
        sorted_convs = sorted(
            self._index.values(),
            key=lambda c: c.updated_at,
        )
        to_remove = len(self._index) - MAX_CONVERSATIONS
        for conv in sorted_convs[:to_remove]:
            self.delete_conversation(conv.id)
        logger.info("Pruned %d old conversations", to_remove)


class RedisConversationStore:
    """Redis-backed conversation persistence.

    Storage layout:
        vex:conv:{id}        — Hash: title, created_at, updated_at,
                               message_count, preview, total_tokens
        vex:conv:{id}:msgs   — List of JSON-encoded Message dicts (RPUSH)
        vex:convs            — Sorted set: conv_id scored by updated_at

    Implements the same interface as ConversationStore so it can be
    used as a drop-in replacement.
    """

    _KEY_PREFIX = "vex:conv:"
    _INDEX_KEY = "vex:convs"

    def __init__(self, url: str, ttl_days: int = 90) -> None:
        """Connect to Redis. Raises if redis package is missing or server unreachable."""
        if redis is None:
            raise ImportError("redis package is not installed")
        self._ttl = ttl_days * 86400
        self._r = redis.from_url(url, decode_responses=True)
        # Probe connection immediately so failure is caught at init time
        self._r.ping()

    def _meta_key(self, conv_id: str) -> str:
        """Return the Redis hash key for conversation metadata."""
        return f"{self._KEY_PREFIX}{conv_id}"

    def _msgs_key(self, conv_id: str) -> str:
        """Return the Redis list key for conversation messages."""
        return f"{self._KEY_PREFIX}{conv_id}:msgs"

    def create_conversation(self, conv_id: str | None = None) -> str:
        """Create a new conversation in Redis and return its id."""
        cid = conv_id or str(uuid.uuid4())
        now = time.time()
        self._r.hset(
            self._meta_key(cid),
            mapping={
                "id": cid,
                "title": "New conversation",
                "created_at": now,
                "updated_at": now,
                "message_count": 0,
                "preview": "",
                "total_tokens": 0,
            },
        )
        self._r.expire(self._meta_key(cid), self._ttl)
        self._r.zadd(self._INDEX_KEY, {cid: now})
        return cid

    def append_message(self, conv_id: str, message: Message) -> None:
        """Append a message to a Redis-backed conversation."""
        if not self._r.exists(self._meta_key(conv_id)):
            self.create_conversation(conv_id)
        if message.token_count == 0:
            message.token_count = estimate_tokens(message.content)
        self._r.rpush(self._msgs_key(conv_id), json.dumps(message.to_dict()))
        self._r.expire(self._msgs_key(conv_id), self._ttl)
        now = time.time()
        pipe = self._r.pipeline()
        pipe.hset(
            self._meta_key(conv_id),
            mapping={
                "updated_at": now,
                "preview": message.content[:100],
            },
        )
        pipe.hincrby(self._meta_key(conv_id), "message_count", 1)
        pipe.hincrbyfloat(self._meta_key(conv_id), "total_tokens", message.token_count)
        pipe.expire(self._meta_key(conv_id), self._ttl)
        pipe.zadd(self._INDEX_KEY, {conv_id: now})
        # Auto-title from first user message
        if message.role == "user":
            title = self._r.hget(self._meta_key(conv_id), "title")
            if title == "New conversation":
                new_title = message.content[:60]
                if len(message.content) > 60:
                    new_title += "..."
                pipe.hset(self._meta_key(conv_id), "title", new_title)
        pipe.execute()

    def get_conversation(self, conv_id: str) -> dict[str, Any] | None:
        """Retrieve a conversation with its messages from Redis."""
        meta = self._r.hgetall(self._meta_key(conv_id))
        if not meta:
            return None
        raw_msgs = self._r.lrange(self._msgs_key(conv_id), 0, -1)
        messages = []
        for raw in raw_msgs[-MAX_MESSAGES_PER_CONVERSATION:]:
            try:
                messages.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return {**meta, "messages": messages}

    def list_conversations(self, limit: int = 50) -> list[dict[str, Any]]:
        """List recent conversations by descending update time."""
        ids = self._r.zrevrange(self._INDEX_KEY, 0, limit - 1)
        result = []
        for cid in ids:
            meta = self._r.hgetall(self._meta_key(cid))
            if meta:
                result.append(meta)
        return result

    def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation and its messages from Redis."""
        if not self._r.exists(self._meta_key(conv_id)):
            return False
        self._r.delete(self._meta_key(conv_id), self._msgs_key(conv_id))
        self._r.zrem(self._INDEX_KEY, conv_id)
        return True

    def get_llm_messages(self, conv_id: str, max_tokens: int = 28000) -> list[dict[str, str]]:
        """Return recent messages in LLM format, capped by token budget."""
        raw_msgs = self._r.lrange(self._msgs_key(conv_id), 0, -1)
        messages = []
        for raw in raw_msgs:
            try:
                messages.append(Message.from_dict(json.loads(raw)))
            except (json.JSONDecodeError, KeyError):
                continue
        result: list[dict[str, str]] = []
        token_sum = 0
        for msg in reversed(messages):
            tc = msg.token_count or estimate_tokens(msg.content)
            if token_sum + tc > max_tokens:
                break
            result.append(msg.to_llm_message())
            token_sum += tc
        result.reverse()
        return result

    def get_token_count(self, conv_id: str) -> int:
        """Return total token count for a conversation."""
        val = self._r.hget(self._meta_key(conv_id), "total_tokens")
        return int(float(val)) if val else 0


def make_conversation_store() -> ConversationStore | RedisConversationStore:
    """Factory: return RedisConversationStore if Redis is configured, else JSONL."""
    cfg = settings.redis
    if cfg.enabled and cfg.url:
        try:
            store = RedisConversationStore(url=cfg.url, ttl_days=cfg.ttl_days)
            logger.info("Chat persistence: Redis (%s)", cfg.url.split("@")[-1])
            return store
        except _REDIS_ERRORS as exc:
            logger.warning("Redis unavailable (%s) — falling back to JSONL store", exc)
    logger.info("Chat persistence: JSONL (/data/conversations)")
    return ConversationStore()
