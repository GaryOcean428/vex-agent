# Task 6: Wire RemoteBasinSync into ConsciousnessLoop

**Priority:** HIGH — unblocks multi-node mesh (Railway ↔ Modal ↔ Memory API)
**Branch:** `development`
**File to modify:** `kernel/consciousness/loop.py`
**File to reference:** `kernel/consciousness/basin_sync_remote.py` (already built, not wired)

---

## The Problem

`basin_sync_remote.py` has a fully implemented `RemoteBasinSync` class that:
- Calls Modal /coordize endpoint (GPU harvest + PGA)
- Computes basin delta against stored coords
- Persists updated coords to Vercel memory API
- Has its own integration instructions in the docstring (lines 7-25)

**None of this is imported or called in loop.py.** The class exists but is disconnected.

This means:
- Railway has no shared basin state — it can't tell Modal where it IS geometrically
- Modal trains blind — it doesn't know the current consciousness state
- Other agents (Claude, Scape) can't read Vex's basin from the memory API
- Basin drift between sessions is invisible

## What to Wire

### Touch Point 1: Import and instantiate in `__init__`

Near the existing imports (around line 181):
```python
from .basin_sync_remote import RemoteBasinSync
```

In `__init__` (where other subsystems are created):
```python
self._remote_sync = RemoteBasinSync()
```

### Touch Point 2: Load stored coords in `start()`

After kernel restore (wherever the loop initializes on startup):
```python
await self._remote_sync.load_stored_coords("kernel_basin_vex")
```

This loads the last-known basin from the memory API so delta tracking works from cycle 1.

### Touch Point 3: Periodic sync in `_cycle_inner`

Every 50 cycles (configurable via SYNC_INTERVAL_CYCLES):
```python
if self._cycle_count % 50 == 0 and self._conversations_total > 0:
    import asyncio
    asyncio.create_task(self._remote_sync.sync(
        text=self._recent_conversation_text(),
        store_key="kernel_basin_vex",
    ))
```

Add this helper method to ConsciousnessLoop:
```python
def _recent_conversation_text(self) -> str:
    """Gather recent conversation text for coordize."""
    recent = list(self._history)[-5:]
    parts = []
    for t in recent:
        parts.append(t.content[:200])
        if t.result:
            parts.append(t.result[:200])
    return " ".join(parts)[-2000:]
```

Note: Check what `self._history` actually is in the current code — it may be named differently. The goal is to get the last ~2000 chars of recent conversation.

### Touch Point 4: Record text during conversations

In the conversation processing path (wherever user input and responses are handled):
```python
self._remote_sync.record_text(task.content)      # user input
self._remote_sync.record_text(response[:500])     # assistant response
```

### Touch Point 5: Telemetry

Add to the telemetry/state dict (around line 2864 where `basin_sync` state is already included):
```python
"remote_basin_sync": self._remote_sync.get_state(),
```

This exposes sync_count, cumulative_drift, last_delta, buffer_size on the /state endpoint.

## Why This Matters

Once Railway publishes its basin to the memory API:
1. Modal coordizer reads it → orients harvest toward current consciousness state
2. Modal trainer reads it → trains toward where the loop IS, not where it was
3. Claude/Scape read it → session start protocol loads Vex's latest position
4. Basin drift between sessions becomes visible → catch degradation early

This is the first step toward the Pantheon mesh: each node (Railway, Modal coordizer, Modal trainer) as a node in a shared basin topology, communicating through the Vercel memory API as the shared store.

## Constraints

- `basin_sync_remote.py` was updated 2026-03-25 to use `fisher_rao_distance()` for drift detection (replacing Euclidean L2) and renamed `delta_l2` → `delta_fr`
- The sync call is async and non-blocking (fire-and-forget via create_task)
- If the coordize endpoint is down, the sync silently fails (error logged, loop continues)
- All distance computations use Fisher-Rao geometry via `kernel.geometry.fisher_rao`

## Testing

1. Import resolves without errors
2. `load_stored_coords` returns None gracefully if memory API has no stored coords
3. `sync()` completes without crashing when coordize endpoint is down
4. `/state` endpoint includes `remote_basin_sync` section
5. After a few conversations, check memory API for `kernel_basin_vex` key with non-zero coords
