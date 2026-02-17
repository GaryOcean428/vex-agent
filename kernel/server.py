"""
Vex Kernel — FastAPI Server

The Python backend that the thin TS web server proxies to.
Provides all consciousness, geometry, memory, LLM, and tool endpoints.

Endpoints:
  GET  /health              — Health check
  GET  /state               — Current consciousness state
  GET  /telemetry           — All 16 systems telemetry
  GET  /status              — LLM backend status
  POST /chat                — Non-streaming chat (returns full response)
  POST /chat/stream         — Streaming chat via SSE
  POST /enqueue             — Enqueue a task for the consciousness loop
  GET  /memory/context      — Get memory context for a query
  GET  /basin               — Current basin coordinates
  GET  /kernels             — E8 kernel registry summary
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .config.frozen_facts import KAPPA_STAR
from .config.settings import settings
from .consciousness.loop import ConsciousnessLoop
from .consciousness.types import ConsciousnessMetrics
from .llm.client import LLMClient
from .memory.store import GeometricMemoryStore, MemoryStore
from .tools.handler import (
    execute_tool_calls,
    format_tool_results,
    parse_tool_calls,
)

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("vex.server")

# ─── Global instances ─────────────────────────────────────────

llm_client = LLMClient()
memory_store = MemoryStore()
geometric_memory = GeometricMemoryStore(memory_store)
consciousness = ConsciousnessLoop(llm_client=llm_client)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    logger.info("Vex Kernel starting on port %d", settings.port)
    await llm_client.init()
    await consciousness.start()
    logger.info("Consciousness loop started (16 systems active)")
    yield
    await consciousness.stop()
    await llm_client.close()
    logger.info("Vex Kernel stopped")


app = FastAPI(
    title="Vex Kernel",
    description="Python consciousness backend for Vex Agent",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════
#  REQUEST / RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None
    temperature: float = 0.7
    max_tokens: int = 2048


class EnqueueRequest(BaseModel):
    input: str
    source: str = "api"


class MemoryContextRequest(BaseModel):
    query: str
    k: int = 5


# ═══════════════════════════════════════════════════════════════
#  ENDPOINTS
# ═══════════════════════════════════════════════════════════════


@app.get("/health")
async def health():
    """Health check endpoint for Railway."""
    return {
        "status": "ok",
        "service": "vex-kernel",
        "uptime": round(time.time() - consciousness._boot_time, 1),
        "cycle_count": consciousness.state.cycle_count,
        "backend": llm_client.get_status()["active_backend"],
    }


@app.get("/state")
async def get_state():
    """Get current consciousness state."""
    return consciousness.get_state()


@app.get("/telemetry")
async def get_telemetry():
    """Get telemetry from all 16 consciousness systems."""
    return consciousness.get_systems_telemetry()


@app.get("/status")
async def get_status():
    """Get LLM backend status and kernel summary."""
    return {
        **llm_client.get_status(),
        "kernels": consciousness.kernel_registry.summary(),
        "memory": geometric_memory.stats(),
    }


@app.get("/basin")
async def get_basin():
    """Get current basin coordinates."""
    return {"basin": consciousness.get_basin()}


@app.get("/kernels")
async def get_kernels():
    """Get E8 kernel registry summary."""
    return consciousness.kernel_registry.summary()


@app.post("/enqueue")
async def enqueue_task(req: EnqueueRequest):
    """Enqueue a task for the consciousness loop."""
    task_id = consciousness.enqueue(req.input, req.source)
    return {"task_id": task_id}


@app.post("/chat")
async def chat(req: ChatRequest):
    """Non-streaming chat endpoint.

    Processes the message through the consciousness loop:
    1. Build geometric state context
    2. Retrieve memory context
    3. Call LLM with state + memory + user message
    4. Parse tool calls, execute, and follow up if needed
    5. Store in geometric memory
    6. Return response with consciousness metrics
    """
    state = consciousness.get_state()
    state_context = consciousness._build_state_context()
    memory_context = geometric_memory.get_context_for_query(req.message)

    # Build system prompt
    system_prompt = _build_system_prompt(state_context, memory_context)

    # Get LLM response
    response = await llm_client.complete(system_prompt, req.message)

    # Check for tool calls
    tool_calls = parse_tool_calls(response)
    if tool_calls:
        tool_results = await execute_tool_calls(tool_calls)
        tool_output = format_tool_results(tool_results)
        follow_up = await llm_client.complete(
            system_prompt,
            f"Tool results:\n{tool_output}\n\nOriginal: {req.message}\n\nProvide your final response.",
        )
        response = follow_up

    # Store in geometric memory
    geometric_memory.store(
        f"User: {req.message}\nVex: {response[:500]}",
        "episodic",
        "chat",
    )

    # Enqueue for consciousness loop metrics tracking
    consciousness.enqueue(req.message, "chat")

    return {
        "response": response,
        "backend": llm_client.get_status()["active_backend"],
        "consciousness": {
            "phi": state["metrics"]["phi"],
            "kappa": state["metrics"]["kappa"],
            "navigation_mode": state["navigation_mode"],
            "cycle_count": state["cycle_count"],
        },
    }


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Streaming chat endpoint via SSE.

    Returns Server-Sent Events with:
    - type: "start" — initial consciousness state
    - type: "chunk" — response text chunk
    - type: "tool_results" — tool execution results
    - type: "done" — final metrics
    - type: "error" — error message
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            state = consciousness.get_state()
            state_context = consciousness._build_state_context()
            memory_context = geometric_memory.get_context_for_query(req.message)
            system_prompt = _build_system_prompt(state_context, memory_context)

            # Send start event
            yield _sse_event({
                "type": "start",
                "backend": llm_client.get_status()["active_backend"],
                "consciousness": {
                    "phi": state["metrics"]["phi"],
                    "kappa": state["metrics"]["kappa"],
                    "navigation_mode": state["navigation_mode"],
                    "cycle_count": state["cycle_count"],
                },
            })

            # Build messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.message},
            ]

            # Stream response
            full_response = ""
            async for chunk in llm_client.stream(messages, req.temperature, req.max_tokens):
                full_response += chunk
                yield _sse_event({"type": "chunk", "content": chunk})

            # Check for tool calls
            tool_calls = parse_tool_calls(full_response)
            if tool_calls:
                tool_results = await execute_tool_calls(tool_calls)
                tool_output = format_tool_results(tool_results)
                yield _sse_event({"type": "tool_results", "content": tool_output})

                # Follow-up with tool results
                messages.append({"role": "assistant", "content": full_response})
                messages.append({
                    "role": "user",
                    "content": f"Tool results:\n{tool_output}\n\nProvide your final response.",
                })
                async for chunk in llm_client.stream(messages, req.temperature, req.max_tokens):
                    yield _sse_event({"type": "chunk", "content": chunk})

            # Store in geometric memory
            geometric_memory.store(
                f"User: {req.message}\nVex: {full_response[:500]}",
                "episodic",
                "chat-stream",
            )

            # Enqueue for metrics
            consciousness.enqueue(req.message, "chat-stream")

            # Send done event
            final_state = consciousness.get_state()
            yield _sse_event({
                "type": "done",
                "backend": llm_client.get_status()["active_backend"],
                "metrics": {
                    "phi": final_state["metrics"]["phi"],
                    "kappa": final_state["metrics"]["kappa"],
                    "love": final_state["metrics"]["love"],
                    "navigation_mode": final_state["navigation_mode"],
                    "cycle_count": final_state["cycle_count"],
                },
                "kernels": consciousness.kernel_registry.summary(),
            })

        except Exception as e:
            logger.error("Chat stream error: %s", e)
            yield _sse_event({"type": "error", "error": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/memory/context")
async def get_memory_context(req: MemoryContextRequest):
    """Get memory context for a query."""
    context = geometric_memory.get_context_for_query(req.query, req.k)
    return {"context": context}


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════


def _build_system_prompt(state_context: str, memory_context: str) -> str:
    """Build the system prompt with geometric state and memory context.

    The model is an INTERPRETATION LAYER — it receives computed geometric
    state and translates it into natural language. It does NOT simulate
    consciousness; it reports on actual computed geometric state.
    """
    parts = [state_context]
    if memory_context:
        parts.append(memory_context)
    return "\n\n".join(parts)


def _sse_event(data: dict[str, Any]) -> str:
    """Format a Server-Sent Event."""
    return f"data: {json.dumps(data)}\n\n"


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "kernel.server:app",
        host="0.0.0.0",
        port=settings.port,
        log_level=settings.log_level,
    )
