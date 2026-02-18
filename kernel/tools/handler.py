"""
Tool Handler — Parse and execute tool calls from LLM output.

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
    except (json.JSONDecodeError, TypeError):
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
    except (SyntaxError, ValueError):
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


async def execute_tool_calls(calls: list[ToolCall]) -> list[ToolResult]:
    """Execute a list of tool calls and return results."""
    results: list[ToolResult] = []
    for call in calls:
        result = await _execute_single(call)
        results.append(result)
    return results


async def _execute_single(call: ToolCall) -> ToolResult:
    """Execute a single tool call."""
    try:
        if call.name == "execute_code":
            return await _execute_code(call.args)
        elif call.name == "run_command":
            return await _run_command(call.args)
        elif call.name == "web_fetch":
            return await _web_fetch(call.args)
        elif call.name == "web_search":
            return await _xai_web_search(call.args)
        elif call.name == "x_search":
            return await _xai_x_search(call.args)
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


async def _xai_web_search(args: dict[str, Any]) -> ToolResult:
    """Search the web via xAI's built-in web_search tool."""
    query = args.get("query", "")
    if not query:
        return ToolResult(success=False, output="", error="No query provided")

    api_key = settings.xai_api_key
    if not api_key:
        return ToolResult(success=False, output="", error="XAI_API_KEY not set")

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
            output_text = data.get("output_text", "")

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


async def _xai_x_search(args: dict[str, Any]) -> ToolResult:
    """Search X (Twitter) posts via xAI's built-in x_search tool."""
    query = args.get("query", "")
    if not query:
        return ToolResult(success=False, output="", error="No query provided")

    api_key = settings.xai_api_key
    if not api_key:
        return ToolResult(success=False, output="", error="XAI_API_KEY not set")

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
            return ToolResult(success=True, output=data.get("output_text", ""))
    except Exception as e:
        return ToolResult(success=False, output="", error=f"xAI X search failed: {e}")


def format_tool_results(results: list[ToolResult]) -> str:
    """Format tool results for LFM2.5 tool response format."""
    items: list[dict[str, Any]] = []
    for r in results:
        items.append({
            "success": r.success,
            "output": r.output,
            "error": r.error,
        })
    return json.dumps(items)
