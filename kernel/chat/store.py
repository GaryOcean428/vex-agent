"""Conversation Store — JSONL-based chat persistence on /data volume.

Provides server-side conversation history so chats survive page
refreshes and Railway redeploys (via mounted volume).

Storage layout:
    {data_dir}/conversations/
        {conversation_id}.jsonl   — one JSON line per message
        _index.json               — conversation list with metadata

Design choices:
  - JSONL for append-only writes (no full-file rewrites per message)
  - Index file for fast conversation listing without scanning all files
  - No database dependency — works on any filesystem
  - Capped at 200 conversations, oldest pruned automatically
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config.settings import settings

logger = logging.getLogger("vex.chat.store")

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
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
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
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Conversation:
        return cls(
            id=data["id"],
            title=data.get("title", "Untitled"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            message_count=data.get("message_count", 0),
            preview=data.get("preview", ""),
            total_tokens=data.get("total_tokens", 0),
        )


class ConversationStore:
    """JSONL-based conversation persistence."""

    def __init__(self, data_dir: str | None = None) -> None:
        self._base_dir = Path(data_dir or settings.data_dir) / "conversations"
        try:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            # Probe writability — mkdir succeeds on existing read-only dirs
            probe = self._base_dir / ".write_probe"
            probe.touch()
            probe.unlink()
        except OSError:
            # Volume not writable — fall back to /tmp (ephemeral but functional)
            self._base_dir = Path("/tmp/vex-conversations")
            self._base_dir.mkdir(parents=True, exist_ok=True)
            logger.warning("Data volume not writable — using ephemeral %s", self._base_dir)
        self._index_path = self._base_dir / "_index.json"
        self._index: dict[str, Conversation] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load the conversation index from disk."""
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
        """Rebuild index by scanning JSONL files."""
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
                    created_at=float(messages[0].timestamp)
                    if messages[0].timestamp.replace(".", "").isdigit()
                    else time.time(),
                    updated_at=float(messages[-1].timestamp)
                    if messages[-1].timestamp.replace(".", "").isdigit()
                    else time.time(),
                    message_count=len(messages),
                    preview=messages[-1].content[:100],
                )
        self._save_index()
        logger.info("Rebuilt conversation index: %d conversations", len(self._index))

    def _save_index(self) -> None:
        """Persist the conversation index."""
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
        """Read all messages for a conversation."""
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
        """Create a new conversation, return its ID."""
        cid = conv_id or str(uuid.uuid4())
        now = time.time()
        self._index[cid] = Conversation(
            id=cid,
            title="New conversation",
            created_at=now,
            updated_at=now,
        )
        # Touch the JSONL file (non-fatal: in-memory index is enough for this session)
        try:
            self._conv_path(cid).touch()
        except OSError as e:
            logger.warning("Could not create conversation file %s: %s", cid, e)
        self._save_index()
        self._prune_old()
        return cid

    def append_message(
        self,
        conv_id: str,
        message: Message,
    ) -> None:
        """Append a message to a conversation."""
        if conv_id not in self._index:
            self.create_conversation(conv_id)

        # Ensure token count is set
        if message.token_count == 0:
            message.token_count = estimate_tokens(message.content)

        path = self._conv_path(conv_id)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(message.to_dict()) + "\n")
        except OSError as e:
            logger.error("Failed to write message to %s: %s", conv_id, e)
            return

        # Update index
        conv = self._index[conv_id]
        conv.updated_at = time.time()
        conv.message_count += 1
        conv.total_tokens += message.token_count
        conv.preview = message.content[:100]

        # Auto-title from first user message
        if message.role == "user" and conv.title == "New conversation":
            conv.title = message.content[:60]
            if len(message.content) > 60:
                conv.title += "..."

        self._save_index()

    def get_conversation(self, conv_id: str) -> dict[str, Any] | None:
        """Get a conversation with all messages."""
        if conv_id not in self._index:
            return None
        conv = self._index[conv_id]
        messages = self._read_messages(conv_id)
        return {
            **conv.to_dict(),
            "messages": [m.to_dict() for m in messages[-MAX_MESSAGES_PER_CONVERSATION:]],
        }

    def list_conversations(self, limit: int = 50) -> list[dict[str, Any]]:
        """List conversations, most recent first."""
        sorted_convs = sorted(
            self._index.values(),
            key=lambda c: c.updated_at,
            reverse=True,
        )
        return [c.to_dict() for c in sorted_convs[:limit]]

    def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation."""
        if conv_id not in self._index:
            return False
        del self._index[conv_id]
        path = self._conv_path(conv_id)
        if path.exists():
            path.unlink()
        self._save_index()
        return True

    def get_llm_messages(self, conv_id: str, max_tokens: int = 28000) -> list[dict[str, str]]:
        """Get conversation messages formatted for LLM, truncated to fit token budget.

        Returns oldest-first messages as [{role, content}]. Drops oldest
        messages if total tokens exceed max_tokens (reserves room for
        system prompt + new user message).
        """
        messages = self._read_messages(conv_id)
        if not messages:
            return []
        result: list[dict[str, str]] = []
        token_sum = 0
        # Walk from newest to oldest, accumulating until budget exhausted
        for msg in reversed(messages):
            tc = msg.token_count or estimate_tokens(msg.content)
            if token_sum + tc > max_tokens:
                break
            result.append(msg.to_llm_message())
            token_sum += tc
        result.reverse()  # Restore chronological order
        return result

    def get_token_count(self, conv_id: str) -> int:
        """Get total token count for a conversation."""
        if conv_id in self._index:
            return self._index[conv_id].total_tokens
        return 0

    def _prune_old(self) -> None:
        """Remove oldest conversations if over the cap."""
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
