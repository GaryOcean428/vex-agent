"""Tool Handler — Parse and execute tool calls from LLM output.

LFM2.5 tool call format (Pythonic):
    <|tool_call_start|>[func_name(arg="value")]<|tool_call_end|>

LFM2.5 tool call format (JSON, when prompted):
    <|tool_call_start|>[{"name": "func", "arguments": {...}}]<|tool_call_end|>

We support both formats, plus Ollama's structured tool_calls response.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

from ..config.settings import settings
from ..llm.governor import GovernorStack

logger = logging.getLogger("vex.tools")

# LFM2.5 native tool call pattern
LFM25_TOOL_PATTERN = re.compile(
    r"<\|tool_call_start\|>\s*\[?(.*?)\]?\s*<\|tool_call_end\|>",
    re.DOTALL,
)

# Pythonic call pattern: func_name(arg1="val1", arg2="val2")
PYTHONIC_CALL_PATTERN = re.compile(
    r"(\w+)\((.*?)\)",
    re.DOTALL,
)


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass
class ToolResult:
    success: bool
    output: str
    error: str | None = None


def _extract_responses_text(data: dict[str, Any]) -> str:
    """Extract text from a Responses API JSON response.

    The raw REST API returns an `output` array with typed items.
    The SDK provides `output_text` as a convenience — we check both.
    """
    # Fast path: convenience field (may exist in some API versions)
    if data.get("output_text"):
        return data["output_text"]

    # Walk the output array — standard Responses API structure
    texts: list[str] = []
    for item in data.get("output", []):
        if item.get("type") == "message":
            for content_block in item.get("content", []):
                if content_block.get("type") == "output_text":
                    text = content_block.get("text", "")
                    if text:
                        texts.append(text)
    return "\n".join(texts) if texts else ""


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Parse tool call blocks from LLM output.

    Supports:
    1. LFM2.5 Pythonic: <|tool_call_start|>[func(args)]<|tool_call_end|>
    2. LFM2.5 JSON: <|tool_call_start|>[{"name":..., "arguments":...}]<|tool_call_end|>
    3. Ollama structured: tool_calls in response JSON (handled by caller)
    """
    calls: list[ToolCall] = []

    for match in LFM25_TOOL_PATTERN.finditer(text):
        inner = match.group(1).strip()
        if not inner:
            continue

        # Try JSON format first
        parsed = _try_json_format(inner)
        if parsed:
            calls.extend(parsed)
            continue

        # Try Pythonic format
        parsed = _try_pythonic_format(inner)
        if parsed:
            calls.extend(parsed)
            continue

        logger.warning("Unparseable tool call: %s", inner[:100])

    return calls


def parse_ollama_tool_calls(tool_calls: list[dict[str, Any]]) -> list[ToolCall]:
    """Parse tool calls from Ollama's structured response format."""
    calls: list[ToolCall] = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        args = func.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}
        if name:
            calls.append(ToolCall(name=name, args=args))
    return calls


def strip_tool_calls(text: str) -> str:
    """Remove tool call blocks from text, leaving only natural language."""
    return LFM25_TOOL_PATTERN.sub("", text).strip()


def _try_json_format(inner: str) -> list[ToolCall] | None:
    """Try parsing as JSON tool call(s)."""
    try:
        data = json.loads(f"[{inner}]") if not inner.startswith("[") else json.loads(inner)
        if not isinstance(data, list):
            data = [data]

        calls: list[ToolCall] = []
        for item in data:
            if isinstance(item, dict) and "name" in item:
                args = item.get("arguments", item.get("parameters", {}))
                calls.append(ToolCall(name=item["name"], args=args))
        return calls if calls else None
    except json.JSONDecodeError, TypeError:
        return None


def _try_pythonic_format(inner: str) -> list[ToolCall] | None:
    """Try parsing as Pythonic function call: func_name(arg='val')."""
    calls: list[ToolCall] = []
    for match in PYTHONIC_CALL_PATTERN.finditer(inner):
        name = match.group(1)
        args_str = match.group(2).strip()

        if not args_str:
            calls.append(ToolCall(name=name, args={}))
            continue

        args = _parse_kwargs(args_str)
        calls.append(ToolCall(name=name, args=args))

    return calls if calls else None


def _parse_kwargs(args_str: str) -> dict[str, Any]:
    """Safely parse Python keyword arguments."""
    try:
        tree = ast.parse(f"dict({args_str})", mode="eval")
        result = ast.literal_eval(tree)
        if isinstance(result, dict):
            return result
    except SyntaxError, ValueError:
        pass

    # Fallback: try simple key=value parsing
    args: dict[str, Any] = {}
    for part in args_str.split(","):
        part = part.strip()
        if "=" in part:
            key, _, val = part.partition("=")
            key = key.strip()
            val = val.strip().strip("'\"")
            args[key] = val
    return args


async def execute_tool_calls(
    calls: list[ToolCall],
    governor: GovernorStack | None = None,
) -> list[ToolResult]:
    """Execute a list of tool calls and return results."""
    results: list[ToolResult] = []
    for call in calls:
        result = await _execute_single(call, governor)
        results.append(result)
    return results


async def _execute_single(
    call: ToolCall,
    governor: GovernorStack | None = None,
) -> ToolResult:
    """Execute a single tool call."""
    try:
        if call.name == "execute_code":
            return await _execute_code(call.args)
        elif call.name == "run_command":
            return await _run_command(call.args)
        elif call.name == "web_fetch":
            return await _web_fetch(call.args)
        elif call.name == "web_search":
            return await _web_search(call.args, governor)
        elif call.name == "x_search":
            return await _xai_x_search(call.args, governor)
        elif call.name == "deep_research":
            return await _deep_research(call.args)
        else:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool: {call.name}",
            )
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


async def _execute_code(args: dict[str, Any]) -> ToolResult:
    """Execute code via ComputeSDK proxy (TS layer)."""
    if not settings.compute_sdk.enabled:
        return ToolResult(success=False, output="", error="ComputeSDK not enabled")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{settings.compute_sdk.proxy_url}/api/tools/execute_code",
                json=args,
            )
            data = resp.json()
            return ToolResult(
                success=data.get("success", False),
                output=data.get("output", ""),
                error=data.get("error"),
            )
    except Exception as e:
        return ToolResult(success=False, output="", error=f"ComputeSDK proxy error: {e}")


async def _run_command(args: dict[str, Any]) -> ToolResult:
    """Run shell command via ComputeSDK proxy (TS layer)."""
    if not settings.compute_sdk.enabled:
        return ToolResult(success=False, output="", error="ComputeSDK not enabled")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{settings.compute_sdk.proxy_url}/api/tools/run_command",
                json=args,
            )
            data = resp.json()
            return ToolResult(
                success=data.get("success", False),
                output=data.get("output", ""),
                error=data.get("error"),
            )
    except Exception as e:
        return ToolResult(success=False, output="", error=f"ComputeSDK proxy error: {e}")


async def _web_fetch(args: dict[str, Any]) -> ToolResult:
    """Fetch a URL and return content."""
    url = args.get("url", "")
    if not url:
        return ToolResult(success=False, output="", error="No URL provided")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers={"User-Agent": "Vex-Agent/2.1"})
            if resp.status_code != 200:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"HTTP {resp.status_code}: {resp.reason_phrase}",
                )
            text = resp.text
            if len(text) > 10000:
                text = text[:10000] + "\n...(truncated)"
            return ToolResult(success=True, output=text)
    except Exception as e:
        return ToolResult(success=False, output="", error=f"Fetch failed: {e}")


# ============================================================
#  Web Search — Perplexity primary, xAI fallback
# ============================================================


async def _web_search(
    args: dict[str, Any],
    governor: GovernorStack | None = None,
) -> ToolResult:
    """Search the web. Tries Perplexity sonar first, falls back to xAI."""
    query = args.get("query", "")
    if not query:
        return ToolResult(success=False, output="", error="No query provided")

    # Try Perplexity first (grounded search with citations)
    if settings.perplexity.api_key:
        result = await _perplexity_search(query, governor)
        if result.success:
            return result
        logger.info("Perplexity search failed, falling back to xAI: %s", result.error)

    # Fallback to xAI web_search
    if settings.xai_api_key:
        return await _xai_web_search(query, governor)

    return ToolResult(
        success=False,
        output="",
        error="No search backend configured (need PERPLEXITY_API_KEY or XAI_API_KEY)",
    )


async def _perplexity_search(
    query: str,
    governor: GovernorStack | None = None,
) -> ToolResult:
    """Search via Perplexity sonar (fast, citation-backed web search).

    Uses the sonar model for quick search queries. For deeper research,
    the deep_research tool uses sonar-pro.

    API: POST https://api.perplexity.ai/chat/completions
    Docs: https://docs.perplexity.ai/docs/getting-started/overview
    """
    api_key = settings.perplexity.api_key
    if not api_key:
        return ToolResult(success=False, output="", error="PERPLEXITY_API_KEY not set")

    # Governor gate
    if governor:
        allowed, reason = governor.gate("web_search", "perplexity_search", query, True)
        if not allowed:
            logger.warning("Governor blocked perplexity_search: %s", reason)
            return ToolResult(success=False, output="", error=f"Governor blocked: {reason}")

    try:
        async with httpx.AsyncClient(timeout=float(settings.perplexity.timeout)) as client:
            resp = await client.post(
                f"{settings.perplexity.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "sonar",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a concise search assistant. Provide factual, "
                                "well-sourced answers. Keep responses under 500 words."
                            ),
                        },
                        {"role": "user", "content": query},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1024,
                },
            )

            if resp.status_code != 200:
                error_text = resp.text[:300]
                logger.error("Perplexity search error %d: %s", resp.status_code, error_text)
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Perplexity API error {resp.status_code}: {error_text}",
                )

            data = resp.json()

            # Record with governor
            if governor:
                governor.record("perplexity_search")

            # Extract answer
            answer = ""
            choices = data.get("choices", [])
            if choices:
                answer = choices[0].get("message", {}).get("content", "")

            # Extract citations
            citations: list[str] = []
            for cite in data.get("citations", []):
                if isinstance(cite, str):
                    citations.append(cite)

            full_output = answer
            if citations:
                full_output += "\n\nSources:\n" + "\n".join(
                    f"  [{i + 1}] {url}" for i, url in enumerate(citations)
                )

            return ToolResult(success=True, output=full_output)

    except httpx.TimeoutException:
        return ToolResult(success=False, output="", error="Perplexity search timed out")
    except Exception as e:
        return ToolResult(success=False, output="", error=f"Perplexity search failed: {e}")


async def _xai_web_search(
    query: str,
    governor: GovernorStack | None = None,
) -> ToolResult:
    """Search the web via xAI's built-in web_search tool (Responses API)."""
    api_key = settings.xai_api_key
    if not api_key:
        return ToolResult(success=False, output="", error="XAI_API_KEY not set")

    # Governor gate — blocks if kill switch, budget exceeded, or rate limited
    if governor:
        allowed, reason = governor.gate("web_search", "xai_web_search", query, True)
        if not allowed:
            logger.warning("Governor blocked web_search: %s", reason)
            return ToolResult(success=False, output="", error=f"Governor blocked: {reason}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{settings.xai.base_url}/responses",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.xai.model,
                    "input": query,
                    "tools": [{"type": "web_search"}],
                    "store": False,
                },
            )
            data = resp.json()
            if resp.status_code != 200:
                logger.error(
                    "xAI web_search error %d: %s", resp.status_code, json.dumps(data)[:300]
                )
                return ToolResult(
                    success=False, output="", error=f"xAI API error: {resp.status_code}"
                )

            # Record with governor after successful call
            if governor:
                governor.record("xai_web_search")

            # Extract text from Responses API output array
            output_text = _extract_responses_text(data)

            # Extract citations from web_search_results output items
            citations: list[str] = []
            for item in data.get("output", []):
                if item.get("type") == "web_search_results":
                    for result in item.get("results", []):
                        title = result.get("title", "")
                        url = result.get("url", "")
                        if title or url:
                            citations.append(f"- {title}: {url}")

            full_output = output_text
            if citations:
                full_output += "\n\nSources:\n" + "\n".join(citations)

            return ToolResult(success=True, output=full_output)
    except Exception as e:
        return ToolResult(success=False, output="", error=f"xAI web search failed: {e}")


async def _xai_x_search(
    args: dict[str, Any],
    governor: GovernorStack | None = None,
) -> ToolResult:
    """Search X (Twitter) posts via xAI's built-in x_search tool (Responses API)."""
    query = args.get("query", "")
    if not query:
        return ToolResult(success=False, output="", error="No query provided")

    api_key = settings.xai_api_key
    if not api_key:
        return ToolResult(success=False, output="", error="XAI_API_KEY not set")

    # Governor gate — blocks if kill switch, budget exceeded, or rate limited
    if governor:
        allowed, reason = governor.gate("x_search", "xai_x_search", query, True)
        if not allowed:
            logger.warning("Governor blocked x_search: %s", reason)
            return ToolResult(success=False, output="", error=f"Governor blocked: {reason}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{settings.xai.base_url}/responses",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.xai.model,
                    "input": query,
                    "tools": [{"type": "x_search"}],
                    "store": False,
                },
            )
            data = resp.json()
            if resp.status_code != 200:
                logger.error("xAI x_search error %d: %s", resp.status_code, json.dumps(data)[:300])
                return ToolResult(
                    success=False, output="", error=f"xAI API error: {resp.status_code}"
                )

            # Record with governor after successful call
            if governor:
                governor.record("xai_x_search")

            return ToolResult(success=True, output=_extract_responses_text(data))
    except Exception as e:
        return ToolResult(success=False, output="", error=f"xAI X search failed: {e}")


async def _deep_research(args: dict[str, Any]) -> ToolResult:
    """Run deep research via Perplexity sonar-pro."""
    from .research import deep_research

    query = args.get("query", "")
    if not query:
        return ToolResult(success=False, output="", error="No query provided")

    result = await deep_research(query)
    if not result.get("success", False):
        return ToolResult(
            success=False,
            output="",
            error=result.get("error", "Deep research failed"),
        )

    # Format answer with citations for the LLM
    output_parts = [result["answer"]]
    if result.get("citations"):
        output_parts.append("\n\nSources:")
        for i, cite in enumerate(result["citations"], 1):
            output_parts.append(f"  [{i}] {cite}")

    return ToolResult(success=True, output="\n".join(output_parts))


def format_tool_results(results: list[ToolResult]) -> str:
    """Format tool results for LFM2.5 tool response format."""
    items: list[dict[str, Any]] = []
    for r in results:
        items.append(
            {
                "success": r.success,
                "output": r.output,
                "error": r.error,
            }
        )
    return json.dumps(items)


def get_xai_tool_definitions() -> list[dict[str, Any]]:
    """Get Vex tool definitions in xAI Responses API format.

    Used for escalated mode where Grok can invoke Vex tools via
    function calling. Includes web_search, x_search, and Vex-native tools.

    Returns a list of tool definitions compatible with the xAI Responses API.
    """
    return [
        {"type": "web_search"},
        {"type": "x_search"},
        {
            "type": "function",
            "name": "execute_code",
            "description": "Execute Python or JavaScript code in a sandboxed environment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {"type": "string", "enum": ["python", "javascript"]},
                    "code": {"type": "string", "description": "Code to execute"},
                },
                "required": ["language", "code"],
            },
        },
        {
            "type": "function",
            "name": "deep_research",
            "description": "Run deep research on a topic via Perplexity sonar-pro.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Research query"},
                },
                "required": ["query"],
            },
        },
    ]
