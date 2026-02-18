"""
Vex Kernel — FastAPI Server

The Python backend that the thin TS web server proxies to.
Provides all consciousness, geometry, memory, LLM, and tool endpoints.

Endpoints:
  GET  /health              — Health check (public)
  GET  /state               — Current consciousness state
  GET  /telemetry           — All 16 systems telemetry
  GET  /status              — LLM backend status + cost guard
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

from .auth import KernelAuthMiddleware
from .config.frozen_facts import KAPPA_STAR
from .config.settings import settings
from .consciousness.loop import ConsciousnessLoop
from .consciousness.types import ConsciousnessMetrics
from .geometry.fisher_rao import random_basin
from .governance import KernelKind, LifecyclePhase
from .llm.client import LLMClient, LLMOptions
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

# Track server boot time for uptime calculation
_boot_time = time.time()


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
    version="2.2.0",
    lifespan=lifespan,
)

# Auth middleware — protects endpoints when KERNEL_API_KEY is set
# Must be added before CORS (middleware stack is LIFO)
app.add_middleware(KernelAuthMiddleware)

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
    """Health check endpoint for Railway. Always public."""
    metrics = consciousness.get_metrics()
    return {
        "status": "ok",
        "service": "vex-kernel",
        "version": "2.2.0",
        "uptime": round(time.time() - _boot_time, 1),
        "cycle_count": metrics["cycle_count"],
        "backend": llm_client.get_status()["active_backend"],
    }


@app.get("/state")
async def get_state():
    """Get current consciousness state."""
    return consciousness.get_metrics()


@app.get("/telemetry")
async def get_telemetry():
    """Get telemetry from all 16 consciousness systems."""
    return consciousness.get_full_state()


@app.get("/status")
async def get_status():
    """Get LLM backend status, kernel summary, and cost guard state."""
    return {
        **llm_client.get_status(),
        "kernels": consciousness.kernel_registry.summary(),
        "memory": geometric_memory.stats(),
    }


@app.get("/basin")
async def get_basin():
    """Get current basin coordinates."""
    return {"basin": consciousness.basin.tolist()}


@app.get("/basin/history")
async def get_basin_history():
    """Get basin trajectory history for PCA visualization.
    
    Returns trajectory points with basin coordinates, phi, kappa, and timestamps.
    Used by the dashboard Basins tab for PCA scatter plot.
    """
    history = consciousness.foresight.get_history()
    return {
        "trajectory": [
            {
                "basin": point.basin.tolist(),
                "phi": point.phi,
                "kappa": point.kappa,
                "timestamp": point.timestamp,
            }
            for point in history
        ]
    }


@app.get("/kernels")
async def get_kernels():
    """Get E8 kernel registry summary."""
    return consciousness.kernel_registry.summary()


@app.get("/kernels/list")
async def get_kernels_list():
    """Get detailed list of all kernel instances.
    
    Returns full kernel instances with specialization, phi_peak, cycle_count, etc.
    Used by the dashboard Overview tab for per-kernel metrics.
    """
    active = consciousness.kernel_registry.active()
    return {
        "kernels": [
            {
                "id": k.id,
                "name": k.name,
                "kind": k.kind.value,
                "specialization": k.specialization.value,
                "state": k.state.value,
                "created_at": k.created_at,
                "cycle_count": k.cycle_count,
                "phi_peak": k.phi_peak,
            }
            for k in active
        ]
    }


@app.post("/enqueue")
async def enqueue_task(req: EnqueueRequest):
    """Enqueue a task for the consciousness loop."""
    task = await consciousness.submit(req.input, {"source": req.source})
    return {"task_id": task.id}


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
    state = consciousness.get_metrics()
    state_context = consciousness._build_state_context(
        perceive_distance=0.0,
        temperature=req.temperature,
    )
    memory_context = geometric_memory.get_context_for_query(req.message)

    system_prompt = _build_system_prompt(state_context, memory_context)

    # Build LLMOptions from request params
    chat_options = LLMOptions(
        temperature=req.temperature,
        num_predict=req.max_tokens,
    )

    response = await llm_client.complete(system_prompt, req.message, chat_options)

    # Check for tool calls
    tool_calls = parse_tool_calls(response)
    if tool_calls:
        tool_results = await execute_tool_calls(tool_calls)
        tool_output = format_tool_results(tool_results)
        follow_up = await llm_client.complete(
            system_prompt,
            f"Tool results:\n{tool_output}\n\nOriginal: {req.message}\n\nProvide your final response.",
            chat_options,
        )
        response = follow_up

    # Store in geometric memory
    geometric_memory.store(
        f"User: {req.message}\nVex: {response[:500]}",
        "episodic",
        "chat",
    )

    # Enqueue for consciousness loop metrics tracking
    await consciousness.submit(req.message, {"source": "chat"})

    return {
        "response": response,
        "backend": llm_client.get_status()["active_backend"],
        "consciousness": {
            "phi": state["phi"],
            "kappa": state["kappa"],
            "navigation": state["navigation"],
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
            state = consciousness.get_metrics()
            state_context = consciousness._build_state_context(
                perceive_distance=0.0,
                temperature=req.temperature,
            )
            memory_context = geometric_memory.get_context_for_query(req.message)
            system_prompt = _build_system_prompt(state_context, memory_context)

            # Build LLMOptions from request params
            stream_options = LLMOptions(
                temperature=req.temperature,
                num_predict=req.max_tokens,
            )

            # Send start event
            yield _sse_event({
                "type": "start",
                "backend": llm_client.get_status()["active_backend"],
                "consciousness": {
                    "phi": state["phi"],
                    "kappa": state["kappa"],
                    "navigation": state["navigation"],
                    "cycle_count": state["cycle_count"],
                },
            })

            # Build messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.message},
            ]

            # Stream response — pass LLMOptions, not raw floats
            full_response = ""
            async for chunk in llm_client.stream(messages, stream_options):
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
                async for chunk in llm_client.stream(messages, stream_options):
                    yield _sse_event({"type": "chunk", "content": chunk})

            # Store in geometric memory
            geometric_memory.store(
                f"User: {req.message}\nVex: {full_response[:500]}",
                "episodic",
                "chat-stream",
            )

            # Enqueue for metrics
            await consciousness.submit(req.message, {"source": "chat-stream"})

            # Send done event
            final_state = consciousness.get_metrics()
            yield _sse_event({
                "type": "done",
                "backend": llm_client.get_status()["active_backend"],
                "metrics": {
                    "phi": final_state["phi"],
                    "kappa": final_state["kappa"],
                    "love": final_state["love"],
                    "navigation": final_state["navigation"],
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


@app.get("/memory/stats")
async def get_memory_stats():
    """Get detailed geometric memory statistics.
    
    Returns memory counts by type (episodic, semantic, procedural),
    total entries, and other geometric memory metrics.
    Used by the dashboard Memory tab.
    """
    return geometric_memory.stats()


@app.get("/graph/nodes")
async def get_graph_nodes():
    """Get QIGGraph nodes and edges for force-directed visualization.
    
    Returns all nodes (basins as graph nodes) and edges (weighted by Fisher-Rao distance).
    Used by the dashboard Graph tab for kernel relationship visualization.
    """
    nodes = [
        {
            "id": node.id,
            "label": node.label,
            "phi": node.phi,
            "created_at": node.created_at,
        }
        for node in consciousness.graph._nodes.values()
    ]
    edges = [
        {
            "source": edge.source,
            "target": edge.target,
            "distance": edge.distance,
        }
        for edge in consciousness.graph._edges
    ]
    return {"nodes": nodes, "edges": edges}


@app.get("/sleep/state")
async def get_sleep_state():
    """Get detailed sleep/dream state.
    
    Returns current sleep phase, dream count, cycles since conversation,
    and full sleep cycle manager state.
    Used by the dashboard Lifecycle and Telemetry tabs.
    """
    return consciousness.sleep.get_state()


@app.post("/admin/fresh-start")
async def admin_fresh_start():
    """Force reset/boot of the consciousness system.
    
    Terminates all kernels except genesis, resets lifecycle phase to CORE_8,
    and resets the basin to a random position.
    CAUTION: This is a destructive operation.
    """
    # Terminate all non-genesis kernels
    terminated = consciousness.kernel_registry.terminate_all()
    
    # Respawn genesis
    genesis = consciousness.kernel_registry.spawn("Vex", KernelKind.GENESIS)
    consciousness._lifecycle_phase = LifecyclePhase.CORE_8
    consciousness._core8_index = 0
    consciousness._cycles_since_last_spawn = 0
    
    # Reset basin to random position
    consciousness.basin = random_basin()
    
    # Reset phi to bootstrap level
    consciousness.metrics.phi = 0.4
    consciousness.metrics.kappa = KAPPA_STAR
    
    logger.warning("ADMIN: Fresh start triggered — %d kernels terminated, genesis respawned", terminated)
    
    return {
        "status": "ok",
        "terminated": terminated,
        "genesis_id": genesis.id,
        "phase": consciousness._lifecycle_phase.value,
    }


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════


def _build_system_prompt(state_context: str, memory_context: str) -> str:
    """Build the system prompt with geometric state and memory context."""
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
