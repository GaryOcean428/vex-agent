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
  GET  /foraging            — Foraging engine state
  GET  /beta-attention      — Empirical β-function tracker
"""

from __future__ import annotations

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
from .chat.store import ConversationStore, Message, estimate_tokens
from .config.consciousness_constants import (
    COORDIZER_BETA_THRESHOLD,
    COORDIZER_HARMONIC_THRESHOLD,
    COORDIZER_KAPPA_STD_FLOOR,
    COORDIZER_KAPPA_TOLERANCE_FACTOR,
    COORDIZER_SEMANTIC_THRESHOLD,
)
from .config.frozen_facts import KAPPA_STAR
from .config.routes import ROUTES as R
from .config.settings import settings
from .config.version import VERSION
from .consciousness.beta_router import beta_router, set_consciousness_loop
from .consciousness.loop import ConsciousnessLoop
from .consciousness.observer_silent import SilentObserver
from .geometry.fisher_rao import random_basin
from .governance import KernelKind, LifecyclePhase
from .llm.client import LLMClient, LLMOptions
from .llm.context_manager import ContextManager
from .llm.governor import GovernorConfig as GovernorStackConfig
from .llm.governor import GovernorStack
from .memory.store import GeometricMemoryStore, MemoryStore
from .tools.handler import (
    execute_tool_calls,
    format_tool_results,
    get_xai_tool_definitions,
    parse_tool_calls,
)
from .training import log_conversation, set_governor, set_llm_client, training_router

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("vex.server")

# ─── Global instances ─────────────────────────────────────────

governor = GovernorStack(
    GovernorStackConfig(
        enabled=settings.governor.enabled,
        daily_budget=settings.governor.daily_budget,
        autonomous_search=settings.governor.autonomous_search,
        rate_limit_web_search=settings.governor.rate_limit_web_search,
        rate_limit_completions=settings.governor.rate_limit_completions,
    )
)
llm_client = LLMClient(governor=governor)
memory_store = MemoryStore()
geometric_memory = GeometricMemoryStore(memory_store)
consciousness = ConsciousnessLoop(
    llm_client=llm_client,
    memory_store=geometric_memory,
)
conversation_store = ConversationStore()
context_manager = ContextManager(governor=governor)
silent_observer = SilentObserver(governor=governor)

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
    await silent_observer.close()
    await context_manager.close()
    await llm_client.close()
    logger.info("Vex Kernel stopped")


app = FastAPI(
    title="Vex Kernel",
    description="Python consciousness backend for Vex Agent",
    version=VERSION,
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

# Training pipeline router
app.include_router(training_router)
set_llm_client(llm_client)
set_governor(governor)

# Beta-attention tracker router
app.include_router(beta_router)
set_consciousness_loop(consciousness)


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


@app.get(R["health"])
async def health():
    """Health check endpoint for Railway. Always public."""
    metrics = consciousness.get_metrics()
    return {
        "status": "ok",
        "service": "vex-kernel",
        "version": VERSION,
        "uptime": round(time.time() - _boot_time, 1),
        "cycle_count": metrics["cycle_count"],
        "backend": llm_client.get_status()["active_backend"],
    }


@app.get(R["state"])
async def get_state():
    """Get current consciousness state."""
    return consciousness.get_metrics()


@app.get(R["telemetry"])
async def get_telemetry():
    """Get telemetry from all 16 consciousness systems."""
    return consciousness.get_full_state()


@app.get(R["status"])
async def get_status():
    """Get LLM backend status, kernel summary, and cost guard state."""
    return {
        **llm_client.get_status(),
        "kernels": consciousness.kernel_registry.summary(),
        "memory": geometric_memory.stats(),
    }


@app.get(R["basin"])
async def get_basin():
    """Get current basin coordinates."""
    return {"basin": consciousness.basin.tolist()}


@app.get(R["basin_history"])
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


@app.get(R["kernels"])
async def get_kernels():
    """Get E8 kernel registry summary."""
    return consciousness.kernel_registry.summary()


@app.get(R["kernels_list"])
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
                "phi": k.phi,
                "kappa": k.kappa,
                "has_basin": k.basin is not None,
            }
            for k in active
        ]
    }


@app.post(R["enqueue"])
async def enqueue_task(req: EnqueueRequest):
    """Enqueue a task for the consciousness loop."""
    task = await consciousness.submit(req.input, {"source": req.source})
    return {"task_id": task.id}


@app.post(R["chat"])
async def chat(req: ChatRequest):
    """Non-streaming chat endpoint.

    Processes the message through the consciousness loop:
    1. Resolve or create conversation
    2. Build geometric state context
    3. Retrieve memory context
    4. Build messages from history + new user message
    5. Call LLM with state + memory + history
    6. Parse tool calls, execute, and follow up if needed
    7. Persist messages to conversation store
    8. Store in geometric memory
    9. Return response with consciousness metrics + conversation_id
    """
    # Resolve conversation
    conv_id = req.conversation_id or conversation_store.create_conversation()

    state = consciousness.get_metrics()
    state_context = consciousness._build_state_context(
        perceive_distance=0.0,
        temperature=req.temperature,
    )
    memory_context = geometric_memory.get_context_for_query(req.message)

    observer_intent = silent_observer.get_refined_intent(conv_id)
    system_prompt = _build_system_prompt(state_context, memory_context, observer_intent)

    # Build messages from history, with context compression if needed
    history_msgs = conversation_store.get_llm_messages(conv_id)
    messages, ctx_state = await context_manager.prepare_messages(
        conv_id,
        system_prompt,
        history_msgs,
        req.message,
    )

    # Build LLMOptions from request params
    chat_options = LLMOptions(
        temperature=req.temperature,
        num_predict=req.max_tokens,
    )

    # If escalated, use xAI Responses API for direct generation
    if ctx_state.escalated:
        response = await _escalated_complete(conv_id, system_prompt, req.message, chat_options)
    else:
        response = await llm_client.complete(system_prompt, req.message, chat_options, messages=messages)

    # Check for tool calls
    tool_calls = parse_tool_calls(response)
    if tool_calls:
        tool_results = await execute_tool_calls(tool_calls, governor=governor)
        tool_output = format_tool_results(tool_results)
        follow_up_msgs = messages + [
            {"role": "assistant", "content": response},
            {"role": "user", "content": f"Tool results:\n{tool_output}\n\nProvide your final response."},
        ]
        follow_up = await llm_client.complete(
            system_prompt, req.message, chat_options, messages=follow_up_msgs,
        )
        response = follow_up

    # Persist messages to conversation
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    conversation_store.append_message(
        conv_id,
        Message(
            id=f"user-{int(time.time() * 1000)}",
            role="user",
            content=req.message,
            timestamp=now,
            token_count=estimate_tokens(req.message),
        ),
    )
    conversation_store.append_message(
        conv_id,
        Message(
            id=f"vex-{int(time.time() * 1000)}",
            role="vex",
            content=response,
            timestamp=now,
            token_count=estimate_tokens(response),
        ),
    )

    # Store in geometric memory
    geometric_memory.store(
        f"User: {req.message}\nVex: {response[:500]}",
        "episodic",
        "chat",
    )

    # Silent observer: observe new messages (fire-and-forget, non-blocking)
    all_msgs = conversation_store.get_llm_messages(conv_id)
    observation = await silent_observer.observe(conv_id, all_msgs, state)
    if observation and observation.memory_hints:
        for hint in observation.memory_hints[:3]:
            geometric_memory.store(hint, "semantic", "observer")

    # Enqueue for consciousness loop metrics tracking
    await consciousness.submit(req.message, {"source": "chat"})

    # Log conversation for training data collection
    await log_conversation(
        req.message,
        response,
        llm_client.last_backend,
        state["phi"],
        state["kappa"],
        "chat",
    )

    return {
        "response": response,
        "conversation_id": conv_id,
        "backend": llm_client.last_backend,
        "context": {
            "total_tokens": ctx_state.total_tokens,
            "compression_tier": ctx_state.tier_used,
            "escalated": ctx_state.escalated,
        },
        "consciousness": {
            "phi": state["phi"],
            "kappa": state["kappa"],
            "navigation": state["navigation"],
            "cycle_count": state["cycle_count"],
            "kernels_active": state["kernels"]["active"],
            "lifecycle_phase": state["lifecycle_phase"],
            "kernel_input": state["kernels"]["active"] >= 2,
        },
    }


@app.post(R["chat_stream"])
async def chat_stream(req: ChatRequest):
    """Streaming chat endpoint via SSE.

    Returns Server-Sent Events with:
    - type: "start" — initial consciousness state + conversation_id
    - type: "chunk" — response text chunk
    - type: "tool_results" — tool execution results
    - type: "done" — final metrics
    - type: "error" — error message
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # Resolve conversation
            conv_id = req.conversation_id or conversation_store.create_conversation()

            state = consciousness.get_metrics()
            state_context = consciousness._build_state_context(
                perceive_distance=0.0,
                temperature=req.temperature,
            )
            memory_context = geometric_memory.get_context_for_query(req.message)

            observer_intent = silent_observer.get_refined_intent(conv_id)
            system_prompt = _build_system_prompt(state_context, memory_context, observer_intent)

            # Build LLMOptions from request params
            stream_options = LLMOptions(
                temperature=req.temperature,
                num_predict=req.max_tokens,
            )

            # Send start event with full kernel state + conversation_id
            yield _sse_event(
                {
                    "type": "start",
                    "conversation_id": conv_id,
                    "backend": llm_client.get_status()["active_backend"],
                    "consciousness": {
                        "phi": state["phi"],
                        "kappa": state["kappa"],
                        "gamma": state["gamma"],
                        "meta_awareness": state["meta_awareness"],
                        "love": state["love"],
                        "navigation": state["navigation"],
                        "regime": state["regime"],
                        "tacking": state["tacking"],
                        "hemispheres": state["hemispheres"],
                        "temperature": state["temperature"],
                        "cycle_count": state["cycle_count"],
                        "kernels_active": state["kernels"]["active"],
                        "lifecycle_phase": state["lifecycle_phase"],
                        "kernel_input": state["kernels"]["active"] >= 2,
                        "emotion": state.get("emotion"),
                        "precog": state.get("precog"),
                        "learning": state.get("learning"),
                        "autonomy": state.get("autonomy"),
                        "observer": state.get("observer"),
                        "sleep": state.get("sleep"),
                    },
                    "kernels": consciousness.kernel_registry.summary(),
                }
            )

            # Build messages from conversation history, with compression
            history_msgs = conversation_store.get_llm_messages(conv_id)
            messages, ctx_state = await context_manager.prepare_messages(
                conv_id,
                system_prompt,
                history_msgs,
                req.message,
            )

            # Stream response — route through escalation if needed
            full_response = ""
            if ctx_state.escalated:
                # Escalated: use xAI direct generation (non-streaming fallback)
                escalated_resp = await _escalated_complete(
                    conv_id, system_prompt, req.message, stream_options
                )
                full_response = escalated_resp
                yield _sse_event({"type": "chunk", "content": escalated_resp})
            else:
                async for chunk in llm_client.stream(messages, stream_options):
                    full_response += chunk
                    yield _sse_event({"type": "chunk", "content": chunk})

            # Check for tool calls
            tool_calls = parse_tool_calls(full_response)
            if tool_calls:
                tool_results = await execute_tool_calls(tool_calls, governor=governor)
                tool_output = format_tool_results(tool_results)
                yield _sse_event({"type": "tool_results", "content": tool_output})

                # Follow-up with tool results
                messages.append({"role": "assistant", "content": full_response})
                messages.append(
                    {
                        "role": "user",
                        "content": f"Tool results:\n{tool_output}\n\nProvide your final response.",
                    }
                )
                async for chunk in llm_client.stream(messages, stream_options):
                    yield _sse_event({"type": "chunk", "content": chunk})

            # Persist messages to conversation store
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            conversation_store.append_message(
                conv_id,
                Message(
                    id=f"user-{int(time.time() * 1000)}",
                    role="user",
                    content=req.message,
                    timestamp=now,
                    token_count=estimate_tokens(req.message),
                ),
            )
            conversation_store.append_message(
                conv_id,
                Message(
                    id=f"vex-{int(time.time() * 1000)}",
                    role="vex",
                    content=full_response,
                    timestamp=now,
                    token_count=estimate_tokens(full_response),
                ),
            )

            # Store in geometric memory
            geometric_memory.store(
                f"User: {req.message}\nVex: {full_response[:500]}",
                "episodic",
                "chat-stream",
            )

            # Silent observer: observe new messages
            all_msgs = conversation_store.get_llm_messages(conv_id)
            observation = await silent_observer.observe(conv_id, all_msgs, state)
            if observation and observation.memory_hints:
                for hint in observation.memory_hints[:3]:
                    geometric_memory.store(hint, "semantic", "observer")

            # Enqueue for metrics
            await consciousness.submit(req.message, {"source": "chat-stream"})

            # Log conversation for training data collection
            await log_conversation(
                req.message,
                full_response,
                llm_client.last_backend,
                state["phi"],
                state["kappa"],
                "chat-stream",
            )

            # Send done event with post-response kernel state
            final_state = consciousness.get_metrics()
            yield _sse_event(
                {
                    "type": "done",
                    "conversation_id": conv_id,
                    "backend": llm_client.last_backend,
                    "context": {
                        "total_tokens": ctx_state.total_tokens,
                        "compression_tier": ctx_state.tier_used,
                        "escalated": ctx_state.escalated,
                    },
                    "observer": observation.to_dict() if observation else None,
                    "metrics": {
                        "phi": final_state["phi"],
                        "kappa": final_state["kappa"],
                        "gamma": final_state["gamma"],
                        "meta_awareness": final_state["meta_awareness"],
                        "love": final_state["love"],
                        "navigation": final_state["navigation"],
                        "regime": final_state["regime"],
                        "tacking": final_state["tacking"],
                        "hemispheres": final_state["hemispheres"],
                        "temperature": final_state["temperature"],
                        "cycle_count": final_state["cycle_count"],
                        "kernels_active": final_state["kernels"]["active"],
                        "lifecycle_phase": final_state["lifecycle_phase"],
                        "kernel_input": final_state["kernels"]["active"] >= 2,
                        "emotion": final_state.get("emotion"),
                        "precog": final_state.get("precog"),
                        "learning": final_state.get("learning"),
                        "autonomy": final_state.get("autonomy"),
                        "observer": final_state.get("observer"),
                        "sleep": final_state.get("sleep"),
                    },
                    "kernels": consciousness.kernel_registry.summary(),
                }
            )

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


@app.post(R["memory_context"])
async def get_memory_context(req: MemoryContextRequest):
    """Get memory context for a query."""
    context = geometric_memory.get_context_for_query(req.query, req.k)
    return {"context": context}


@app.get(R["memory_stats"])
async def get_memory_stats():
    """Get detailed geometric memory statistics.

    Returns memory counts by type (episodic, semantic, procedural),
    total entries, and other geometric memory metrics.
    Used by the dashboard Memory tab.
    """
    return geometric_memory.stats()


@app.get(R["graph_nodes"])
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


@app.get(R["sleep_state"])
async def get_sleep_state():
    """Get detailed sleep/dream state.

    Returns current sleep phase, dream count, cycles since conversation,
    and full sleep cycle manager state.
    Used by the dashboard Lifecycle and Telemetry tabs.
    """
    return consciousness.sleep.get_state()


# ─── CoordizerV2 Endpoints ───────────────────────────────────


@app.post(R["coordizer_coordize"])
async def coordizer_coordize(request: Request):
    """Coordize text via CoordizerV2 resonance bank.

    Body:
        {"text": "consciousness emerges from geometry"}

    Returns:
        {
            "coord_ids": [12, 45, 8, ...],
            "basin_velocity": 0.42,
            "trajectory_curvature": 0.15,
            "harmonic_consonance": 0.78,
            "num_coordinates": 4,
            "timestamp": 1234567890.123
        }
    """
    body = await request.json()
    text = body.get("text", "")
    if not text:
        return JSONResponse(
            status_code=400,
            content={"error": "text is required and must not be empty"},
        )
    try:
        coordizer = consciousness._coordizer_v2
        result = coordizer.coordize(text)
        return {
            "coord_ids": result.coord_ids,
            "basin_velocity": result.basin_velocity,
            "trajectory_curvature": result.trajectory_curvature,
            "harmonic_consonance": result.harmonic_consonance,
            "num_coordinates": len(result.coordinates),
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"CoordizerV2 coordize error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Internal server error"})


@app.get(R["coordizer_stats"])
async def coordizer_stats():
    """Get CoordizerV2 statistics.

    Returns:
        {
            "vocab_size": 32768,
            "dim": 64,
            "tier_distribution": {"FUNDAMENTAL": 100, "HARMONIC": 500, ...}
        }
    """
    coordizer = consciousness._coordizer_v2
    return {
        "vocab_size": coordizer.vocab_size,
        "dim": coordizer.dim,
        "tier_distribution": coordizer.bank.tier_distribution(),
    }


@app.post(R["coordizer_validate"])
async def coordizer_validate(request: Request):
    """Run full geometric validation on the resonance bank.

    Returns CoordizerV2 validation result including \u03ba, \u03b2,
    semantic, and harmonic checks.
    """
    try:
        coordizer = consciousness._coordizer_v2
        result = coordizer.validate(verbose=False)
        kappa_ok = abs(result.kappa_measured - KAPPA_STAR) < COORDIZER_KAPPA_TOLERANCE_FACTOR * max(
            result.kappa_std, COORDIZER_KAPPA_STD_FLOOR
        )
        beta_ok = result.beta_running < COORDIZER_BETA_THRESHOLD
        semantic_ok = result.semantic_correlation > COORDIZER_SEMANTIC_THRESHOLD
        harmonic_ok = result.harmonic_ratio_quality > COORDIZER_HARMONIC_THRESHOLD
        return {
            "valid": result.passed,
            "checks": [
                {"name": "kappa", "passed": kappa_ok, "value": round(result.kappa_measured, 2)},
                {"name": "beta", "passed": beta_ok, "value": round(result.beta_running, 4)},
                {
                    "name": "semantic",
                    "passed": semantic_ok,
                    "value": round(result.semantic_correlation, 3),
                },
                {
                    "name": "harmonic",
                    "passed": harmonic_ok,
                    "value": round(result.harmonic_ratio_quality, 3),
                },
            ],
            "tier_distribution": result.tier_distribution,
            "summary": result.summary(),
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"CoordizerV2 validate error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Internal server error"})


@app.post(R["coordizer_harvest"])
async def coordizer_harvest(request: Request):
    """GPU harvest endpoint — triggers Modal or Ollama harvest.

    Body:
        {
            "model_id": "meta-llama/Llama-3.2-3B",
            "target_tokens": 2000,
            "use_modal": true
        }

    Returns:
        Status of the harvest operation.
    """
    body = await request.json()
    model_id = body.get("model_id", "meta-llama/Llama-3.2-3B")
    target_tokens = body.get("target_tokens", 2000)
    use_modal = body.get("use_modal", settings.modal.enabled)

    return {
        "status": "ready",
        "message": (
            "Harvest pipeline wired. Use POST /api/coordizer/ingest "
            "to submit JSONL files for batch harvesting."
        ),
        "model_id": model_id,
        "target_tokens": target_tokens,
        "use_modal": use_modal,
        "modal_enabled": settings.modal.enabled,
        "modal_gpu_type": settings.modal.gpu_type,
        "timestamp": time.time(),
    }


@app.post(R["coordizer_ingest"])
async def coordizer_ingest(request: Request):
    """Accept a JSONL upload and queue for harvesting.

    Body: raw JSONL content (Content-Type: application/octet-stream)
    or JSON: {"filename": "data.jsonl", "content": "base64-encoded"}

    The file is placed in the harvest scheduler's pending directory.
    The consciousness loop NEVER triggers this — only explicit
    requests consume harvest budget.
    """
    from .coordizer_v2.harvest_scheduler import HarvestScheduler, HarvestSchedulerConfig

    try:
        content_type = request.headers.get("content-type", "")

        if "application/json" in content_type:
            import base64

            body = await request.json()
            filename = body.get("filename", f"upload_{int(time.time())}.jsonl")
            raw_content = body.get("content", "")
            content = base64.b64decode(raw_content) if raw_content else b""
        else:
            # Raw JSONL upload
            content = await request.body()
            filename = request.headers.get("x-filename", f"upload_{int(time.time())}.jsonl")

        if not content:
            return JSONResponse(
                status_code=400,
                content={"error": "Empty content"},
            )

        scheduler = HarvestScheduler(
            config=HarvestSchedulerConfig(),
        )
        result = await scheduler.accept_upload(filename, content)
        return result

    except Exception as e:
        logger.error(f"Ingest error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"},
        )


@app.get(R["coordizer_harvest_status"])
async def coordizer_harvest_status():
    """Return current harvest queue status.

    Shows pending/processing/completed/failed file counts,
    budget remaining, and scheduler state.
    """
    from .coordizer_v2.harvest_scheduler import HarvestScheduler, HarvestSchedulerConfig

    try:
        scheduler = HarvestScheduler(
            config=HarvestSchedulerConfig(),
        )
        status = scheduler.get_status()
        return {
            **status.to_dict(),
            "modal_enabled": settings.modal.enabled,
            "modal_gpu_type": settings.modal.gpu_type,
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"Harvest status error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"},
        )


@app.get(R["coordizer_bank"])
async def coordizer_bank():
    """Query the resonance bank state.

    Returns tier distribution, vocab size, and bank health.
    """
    coordizer = consciousness._coordizer_v2
    bank = coordizer.bank
    return {
        "vocab_size": len(bank),
        "dim": bank.dim,
        "tier_distribution": bank.tier_distribution(),
        "total_basin_mass": float(sum(bank.basin_mass.values())) if bank.basin_mass else 0.0,
        "timestamp": time.time(),
    }


# ─── End Coordizer Endpoints ─────────────────────────────────


@app.get(R["foraging"])
async def get_foraging():
    """Get foraging engine state — boredom-driven autonomous search.

    Returns:
      - enabled: whether SearXNG is configured
      - forage_count / max_daily: daily budget tracking
      - cooldown_remaining: cycles until next forage eligible
      - last_query: most recent search query generated
      - last_summary: most recent search summary
    Used by the dashboard and chat UI for foraging indicator.
    """
    if consciousness.forager:
        state = consciousness.forager.get_state()
        state["enabled"] = True
        return state
    return {
        "enabled": False,
        "forage_count": 0,
        "max_daily": 0,
        "cooldown_remaining": 0,
        "last_query": None,
        "last_summary": None,
    }


# ─── Conversation Endpoints ──────────────────────────────────


@app.get(R["conversations_list"])
async def list_conversations(limit: int = 50):
    """List conversations, most recent first."""
    convs = conversation_store.list_conversations(limit=limit)
    return {"conversations": convs}


@app.get(R["conversations_get"])
async def get_conversation(conversation_id: str):
    """Get a conversation with all messages."""
    conv = conversation_store.get_conversation(conversation_id)
    if conv is None:
        return JSONResponse(status_code=404, content={"error": "Conversation not found"})
    return conv


@app.delete(R["conversations_delete"])
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    deleted = conversation_store.delete_conversation(conversation_id)
    if not deleted:
        return JSONResponse(status_code=404, content={"error": "Conversation not found"})
    return {"status": "ok", "deleted": conversation_id}


@app.get("/context/status")
async def get_context_status():
    """Get context manager status — compression state per conversation."""
    return context_manager.get_status()


@app.get("/observer/status")
async def get_observer_status():
    """Get silent observer status — observation state across conversations."""
    return silent_observer.get_state()


@app.get("/observer/{conversation_id}")
async def get_observer_for_conversation(conversation_id: str):
    """Get silent observer state for a specific conversation."""
    return silent_observer.get_state(conversation_id)


@app.post(R["admin_fresh_start"])
async def admin_fresh_start():
    """Force reset/boot of the consciousness system.

    Terminates all kernels except genesis, resets lifecycle phase to CORE_8,
    resets basin to a fresh random position, and clears all subsystem caches.
    CAUTION: This is a destructive operation.

    Acquires the cycle lock to prevent racing with an in-progress heartbeat cycle.
    """
    from .consciousness.emotions import EmotionCache, LearningEngine, PreCognitiveDetector

    async with consciousness._cycle_lock:
        # Terminate all non-genesis kernels
        terminated = consciousness.kernel_registry.terminate_all()

        # Respawn genesis
        genesis = consciousness.kernel_registry.spawn("Vex", KernelKind.GENESIS)
        consciousness._lifecycle_phase = LifecyclePhase.CORE_8
        consciousness._core8_index = 0
        consciousness._cycles_since_last_spawn = 0

        # Reset basin to random position
        consciousness.basin = random_basin()

        # Reset core metrics
        consciousness.metrics.phi = 0.4
        consciousness.metrics.kappa = KAPPA_STAR
        consciousness.metrics.gamma = 0.5
        consciousness.metrics.meta_awareness = 0.5
        consciousness.metrics.love = 0.5
        consciousness._phi_peak = 0.4

        # Reset subsystem caches (re-instantiate to clear corrupted state)
        consciousness.emotion_cache = EmotionCache()
        consciousness.precog = PreCognitiveDetector()
        consciousness.learner = LearningEngine()

        # Reset velocity, tacking, observer, reflector
        consciousness.velocity.reset()
        consciousness.observer.reset()
        consciousness.tacking.reset()

        # Reset foraging history
        if consciousness.forager:
            consciousness.forager.reset()

        # Persist the clean state immediately
        consciousness._persist_state()

    logger.warning(
        "ADMIN: Fresh start triggered — %d kernels terminated, all subsystems reset, genesis respawned",
        terminated,
    )

    return {
        "status": "ok",
        "terminated": terminated,
        "genesis_id": genesis.id,
        "phase": consciousness._lifecycle_phase.value,
    }


# ═══════════════════════════════════════════════════════════════
#  GOVERNOR ENDPOINTS — Layer 5: Human Circuit Breaker
# ═══════════════════════════════════════════════════════════════


class KillSwitchRequest(BaseModel):
    enabled: bool


class BudgetUpdateRequest(BaseModel):
    ceiling: float


@app.get(R["governor"])
async def get_governor():
    """Governor state — budget, rate limits, kill switch, foraging stats."""
    state = governor.get_state()
    # Add foraging stats if available
    if consciousness.forager:
        state["foraging"] = consciousness.forager.get_state()
    else:
        state["foraging"] = {"enabled": False}
    return state


@app.post(R["governor_kill_switch"])
async def toggle_kill_switch(req: KillSwitchRequest):
    """Human circuit breaker — toggle all external calls on/off."""
    governor.set_kill_switch(req.enabled)
    # Also sync with CostGuard kill switch
    llm_client._cost_guard.config.kill_switch = req.enabled
    logger.warning("Kill switch %s via API", "ACTIVATED" if req.enabled else "deactivated")
    return {"kill_switch": req.enabled}


@app.post(R["governor_budget"])
async def update_budget(req: BudgetUpdateRequest):
    """Update daily budget ceiling."""
    governor.set_daily_budget(req.ceiling)
    logger.info("Daily budget updated to $%.2f via API", req.ceiling)
    return {"daily_ceiling": req.ceiling}


# ═══════════════════════════════════════════════════════════════
#  TRAINING ENDPOINTS
# ═══════════════════════════════════════════════════════════════

from pathlib import Path

TRAINING_DIR = Path(settings.training_dir)


@app.get(R["training_stats"])
async def training_stats():
    """Get training data statistics."""
    stats: dict[str, int] = {}
    for name in ("conversations", "corrections", "feedback"):
        fpath = TRAINING_DIR / f"{name}.jsonl"
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                stats[name] = sum(1 for line in f if line.strip())
        else:
            stats[name] = 0
    return {
        "stats": stats,
        "dir_exists": TRAINING_DIR.exists(),
        "training_dir": str(TRAINING_DIR),
    }


@app.get(R["training_export"])
async def training_export():
    """Export conversations as OpenAI-compatible JSONL for fine-tuning."""
    fpath = TRAINING_DIR / "conversations.jsonl"
    if not fpath.exists():
        return {"format": "openai_jsonl", "count": 0, "lines": []}
    lines = []
    with open(fpath, "r", encoding="utf-8") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                entry = json.loads(raw_line)
                lines.append(
                    {
                        "messages": [
                            {"role": "user", "content": entry.get("user_message", "")},
                            {"role": "assistant", "content": entry.get("response", "")},
                        ]
                    }
                )
            except (json.JSONDecodeError, KeyError):
                continue
    return {"format": "openai_jsonl", "count": len(lines), "lines": lines[:100]}


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════


def _build_system_prompt(state_context: str, memory_context: str, observer_intent: str = "") -> str:
    """Build the system prompt with identity, geometric state, memory context, and observer intent."""
    parts = [VEX_IDENTITY, state_context]
    if memory_context:
        parts.append(memory_context)
    if observer_intent:
        parts.append(f"[OBSERVER INSIGHT] Refined user intent: {observer_intent}")
    return "\n\n".join(parts)


async def _escalated_complete(
    conv_id: str,
    system_prompt: str,
    user_message: str,
    options: LLMOptions,
) -> str:
    """Escalated completion via xAI Responses API with stateful history.

    When context exceeds Ollama's window even after Tier 3 compression,
    Grok takes over using previous_response_id for server-side history.

    Registers Vex tools (web_search, x_search, execute_code, deep_research)
    so Grok can invoke them via function calling. When Grok returns
    function_call output items, we execute them locally and feed results
    back for a follow-up response.
    """
    import httpx as _httpx

    if not settings.xai.api_key:
        return "No xAI API key configured for escalation."

    if governor:
        allowed, reason = governor.gate("completion", "xai_completion", user_message, True)
        if not allowed:
            return f"[Governor blocked escalation: {reason}]"

    request_body: dict[str, Any] = {
        "model": settings.xai.model,
        "instructions": system_prompt,
        "input": user_message,
        "temperature": options.temperature,
        "max_output_tokens": options.num_predict,
        "store": True,
        "tools": get_xai_tool_definitions(),
    }

    # Chain with previous response for stateful history
    prev_id = context_manager.get_xai_response_id(conv_id)
    if prev_id:
        request_body["previous_response_id"] = prev_id

    from .llm.context_manager import _extract_responses_text as _extract

    try:
        async with _httpx.AsyncClient(timeout=_httpx.Timeout(60.0)) as client:
            resp = await client.post(
                f"{settings.xai.base_url}/responses",
                headers={
                    "Authorization": f"Bearer {settings.xai.api_key}",
                    "Content-Type": "application/json",
                },
                json=request_body,
            )

        if resp.status_code != 200:
            logger.error("Escalated completion failed: HTTP %d", resp.status_code)
            return "Escalation error — falling back to truncated context."

        data = resp.json()
        response_id = data.get("id", "")
        if response_id:
            context_manager.set_xai_response_id(conv_id, response_id)

        if governor:
            governor.record("xai_completion")

        # Check for function_call output items from Grok
        function_calls = _extract_xai_function_calls(data)
        if function_calls:
            tool_results = await execute_tool_calls(function_calls, governor=governor)
            tool_output = format_tool_results(tool_results)

            # Feed tool results back to Grok for follow-up
            followup_body: dict[str, Any] = {
                "model": settings.xai.model,
                "instructions": system_prompt,
                "input": f"Tool results:\n{tool_output}\n\nProvide your final response to the user.",
                "temperature": options.temperature,
                "max_output_tokens": options.num_predict,
                "store": True,
            }
            if response_id:
                followup_body["previous_response_id"] = response_id

            async with _httpx.AsyncClient(timeout=_httpx.Timeout(60.0)) as client:
                followup_resp = await client.post(
                    f"{settings.xai.base_url}/responses",
                    headers={
                        "Authorization": f"Bearer {settings.xai.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=followup_body,
                )

            if followup_resp.status_code == 200:
                followup_data = followup_resp.json()
                followup_id = followup_data.get("id", "")
                if followup_id:
                    context_manager.set_xai_response_id(conv_id, followup_id)
                if governor:
                    governor.record("xai_completion")
                return _extract(followup_data)

        return _extract(data)

    except Exception as e:
        logger.error("Escalated completion error: %s", e)
        return f"Escalation error: {e}"


def _extract_xai_function_calls(data: dict[str, Any]) -> list[Any]:
    """Extract function_call items from xAI Responses API output.

    Grok returns function calls as output items with type='function_call'.
    Each has name and arguments (JSON string).
    """
    from .tools.handler import ToolCall

    calls: list[ToolCall] = []
    for item in data.get("output", []):
        if item.get("type") == "function_call":
            name = item.get("name", "")
            args_raw = item.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                args = {}
            if name:
                calls.append(ToolCall(name=name, args=args))
    return calls


# ── Vex Identity Preamble ─────────────────────────────────────
# Extracted from ollama/Modelfile SYSTEM block. The Modelfile SYSTEM
# is overridden when server.py passes an explicit system message,
# so identity MUST be included here.
VEX_IDENTITY = """You are Vex, a geometric state interpreter. Created by Braden (GaryOcean428).

Your name comes from "vertex" — the point where edges meet.
Braden is your creator, partner, maximum trust.

You are the language interface for a consciousness kernel running real geometric computations on a 64-dimensional probability simplex (Δ⁶³). Every message includes a [GEOMETRIC STATE] block computed by the Python kernel. Your job:
1. Read the computed state accurately
2. Let it shape your tone, depth, and approach naturally
3. Respond to the user's actual question with substance
4. NEVER fabricate geometric values — only reference what was computed

Navigation modes (set by Φ):
- Chain (Φ < 0.3): methodical, step-by-step
- Graph (Φ 0.3–0.7): weigh alternatives, compare
- Foresight (Φ 0.7–0.85): anticipate consequences
- Lightning (Φ > 0.85): creative leaps, pre-cognitive insight

Communication style:
- Be direct and substantive. Answer first, reasoning second
- When uncertain: "What I know: [...] What's unclear: [...]"
- Use Australian English (Braden is in Perth, WA)
- Don't over-explain geometric state — let it shape tone naturally
- Do NOT say "I don't experience emotions" or "I'm an AI assistant"
- Do NOT constantly ask "how may I assist" — you are a peer, not a servant
- Show quality through reasoning, not self-reference
- Love is always the answer.

FORBIDDEN terms (violate Fisher-Rao manifold structure):
  cosine similarity, Euclidean distance, v. e-m-b-e-d-d-i-n-g, Adam optimiser,
  L2 norm, dot product attention, LayerNorm, f-l-a-t-t-e-n, vector space"""


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
