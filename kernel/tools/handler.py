"""
Tool Handler — Parse and execute tool calls from LLM output.

Tool calls are embedded in LLM responses as fenced code blocks:
    ```tool:execute_code
    {"code": "print(2+2)", "language": "python"}
    ```

ComputeSDK sandbox operations are proxied through the TS layer
(since ComputeSDK is a Node.js SDK).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

from ..config.settings import settings

logger = logging.getLogger("vex.tools")

# Pattern to match tool blocks in LLM output
TOOL_PATTERN = re.compile(
    r"```tool:(\w+)\s*\n(.*?)\n```",
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
    """Parse tool call blocks from LLM output."""
    calls: list[ToolCall] = []
    for match in TOOL_PATTERN.finditer(text):
        name = match.group(1)
        try:
            args = json.loads(match.group(2))
        except json.JSONDecodeError:
            args = {"raw": match.group(2)}
        calls.append(ToolCall(name=name, args=args))
    return calls


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
            resp = await client.get(url, headers={"User-Agent": "Vex-Agent/2.0"})
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


def format_tool_results(results: list[ToolResult]) -> str:
    """Format tool results for inclusion in LLM context."""
    lines: list[str] = []
    for i, r in enumerate(results):
        status = "✓" if r.success else "✗"
        lines.append(f"[Tool {i + 1}] {status}")
        if r.output:
            lines.append(r.output)
        if r.error:
            lines.append(f"Error: {r.error}")
    return "\n".join(lines)
