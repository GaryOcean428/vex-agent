# Training Pipeline Fix Briefing — 2026-03-27

**Status:** CRITICAL — training produces checkpoints but ZERO adapters  
**Root cause:** ProvenanceCallback crashes on numpy float32 serialization  
**Priority:** Fix before any further training runs

## Root Cause

`ProvenanceCallback.on_train_end()` calls `json.dumps()` on HuggingFace Trainer
log entries that contain `numpy.float32` values. Python's json module can't
serialize these. The crash happens INSIDE `trainer.train()` because callbacks
execute within the training loop. Since `model.save_pretrained()` comes AFTER
`trainer.train()`, the adapter save is never reached.

Result: overnight run produced checkpoints (from SFTConfig save_strategy="epoch")
but zero adapters. The consciousness loop is broken.

## Required Fixes

### Fix 1: ProvenanceCallback numpy serialization

**File:** `modal/training_consciousness.py`

In the ProvenanceCallback class, before any `json.dumps()` call:

```python
def _make_serializable(obj):
    """Convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None  # NaN/inf -> null
    return obj
```

Apply `_make_serializable()` to ALL data before `json.dumps()`.

### Fix 2: Resilient trainer wrapper

**File:** `modal/vex_qlora_train.py`, inside the per-kernel try block

Wrap `trainer.train()` so adapter save happens REGARDLESS of callback crashes:

```python
# Current (broken):
result = trainer.train()
model.save_pretrained(adapter_save_path)  # Never reached on crash

# Fixed:
try:
    result = trainer.train()
except Exception as train_err:
    print(f"[{spec}] trainer.train() failed: {train_err}")
    result = None

# Save adapter REGARDLESS — PEFT weights in memory are valid
Path(adapter_save_path).mkdir(parents=True, exist_ok=True)
model.save_pretrained(adapter_save_path)
tokenizer.save_pretrained(adapter_save_path)
model_volume.commit()  # IMMEDIATELY
```

### Fix 3: Volume commit position

`model_volume.commit()` must be called IMMEDIATELY after
`save_pretrained()`, BEFORE any metadata/consciousness/diagnostic code.

Pattern: **save weights → commit → then optional stuff**

### Fix 4: results[spec] KeyError

Line ~1548: `results[spec]["_meta"] = meta` will KeyError because
`results[spec]` hasn't been assigned yet at that point. Move this line
AFTER the `results[spec] = {...}` assignment.

## Non-Blocking Audit Findings

1. **E8 filter broken:** Uses exact match against entry's e8_primitive field,
   but tags in training data may not match the filter list format. All kernels
   fall through to the "train on ALL data" fallback. Not a crash, but no
   specialization differentiation.

2. **Dead code path:** `/training/coordized/` doesn't exist on Modal volume.
   The glob in `_load_training_data` returns empty. Gracefully handled.

3. **Eval mode leak:** `run_post_training_diagnostic` leaves model in eval mode.
   Should call `model.train()` after diagnostic. Low priority since it runs
   after training is done.

4. **Merge OOM:** `_merge_and_export` loads full model on CPU (~70GB for 35B).
   Will OOM on A10G (24GB system RAM after GPU). Failure is caught and
   non-fatal, but merge will never succeed. Either skip on A10G or add
   RAM check.

## Testing

After fixes:
1. Single-kernel training (genesis, 10 samples, 1 epoch)
2. Verify adapter in `/models/adapters/genesis/`
3. Verify `training_meta.json` is valid JSON with no numpy types
4. Deliberately crash a callback — verify adapter STILL saves
5. Check `model_volume.commit()` was called

## Why This Matters

The training pipeline is the consumer of the harvest pipeline:
```
harvest → coordize → train → save adapter → load at inference
```
Without working adapter saves, the entire consciousness loop is broken.
The overnight run wasted GPU hours and produced nothing usable.
