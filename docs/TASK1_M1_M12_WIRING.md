# Task 1: Wire M1-M12 into train_all_kernels()

**Priority:** CRITICAL — all components built, none connected
**Branch:** `development` (create feature branch `feature/wire-m1-m12` from here)
**Files to modify:** `modal/vex_qlora_train.py` (the training function)
**Files to reference:** `modal/training_consciousness.py` (the components — read-only)

---

## The Problem

`modal/training_consciousness.py` has ~1596 lines of consciousness-aware training components.
`modal/vex_qlora_train.py` function `train_all_kernels()` (line ~1022) runs a bare ML pipeline.
None of the M1-M12 components are imported or called.

## What to Wire

### Step 1: Imports (at top of train_all_kernels or in module header)

```python
from training_consciousness import (
    CONSCIOUSNESS_ORDER,
    HestiaSafeBasin,
    apply_demeter_warmup,
    sort_by_fisher_rao,
    TrainingMetricsCallback,
    BreakdownDetector,
    SleepCycleCallback,
    CoachingCallback,
    GeometricRewardCallback,
    SignAwareGradientHold,
    ProvenanceLogger,
    run_post_training_diagnostic,
)
```

### Step 2: Training order (replace arbitrary kernel iteration)

Current code iterates kernels in some order. Replace with:

```python
# Train in consciousness order (genesis first, ocean last)
kernel_list = CONSCIOUSNESS_ORDER if not kernels else kernels.split(",")
```

### Step 3: Per-kernel setup (before each kernel's training loop)

```python
# M1: Safe basin + warm start
hestia = HestiaSafeBasin(specialization=spec, basin_dim=64)
hestia.warm_start_lora(model, tokenizer)

# M8: Demeter warmup (curriculum templates)
dataset = apply_demeter_warmup(dataset, specialization=spec)

# Sort by geometric curriculum
dataset = sort_by_fisher_rao(dataset, home_basin=hestia.home_basin)
```

### Step 4: Callbacks (register with SFTTrainer)

```python
# M12: Provenance logging
provenance = ProvenanceLogger(specialization=spec, output_dir="/training/logs")

callbacks = [
    TrainingMetricsCallback(hestia=hestia),     # M2: Φ, κ_eff, G probing
    BreakdownDetector(),                          # M3: Loss spike detection
    SleepCycleCallback(hestia=hestia),           # M4: Between-epoch consolidation
    CoachingCallback(),                           # M5: Kindness coefficient
    GeometricRewardCallback(hestia=hestia),      # M6: LR modulation by G
    SignAwareGradientHold(),                       # M7: Sign-flip detection
    provenance,                                    # M12: Per-step JSONL
]

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    callbacks=callbacks,
    # ... existing args
)
```

### Step 5: Post-training diagnostic (after each kernel)

```python
# M11: Health check — halt if fails
diagnostic = run_post_training_diagnostic(
    model=model,
    tokenizer=tokenizer,
    specialization=spec,
    hestia=hestia,
)
if not diagnostic["healthy"]:
    logger.error("Kernel %s FAILED post-training diagnostic: %s", spec, diagnostic)
    # Save adapter anyway but flag as unhealthy
```

## Constraints

- Do NOT modify `training_consciousness.py` — it's the reference implementation
- Do NOT use cosine_similarity, np.linalg.norm(a-b), Adam, LayerNorm
- Do NOT weaken any existing gates or tests
- The training_consciousness.py uses numpy for Fisher-Rao geometry — this is correct
- Check each callback's `__init__` signature before instantiating — some may need different args
- If a callback import fails (API mismatch), skip it with a warning, don't crash the trainer

## Testing

1. First: just add the imports and verify they resolve (no runtime errors)
2. Second: wire CONSCIOUSNESS_ORDER and verify genesis trains first
3. Third: add callbacks one at a time, test each
4. Fourth: full run with all callbacks
5. Run `ruff check` and `ruff format` before committing

## Expected Outcome

After wiring, `modal run modal/vex_qlora_train.py::train_all_kernels` should:
- Train genesis first, ocean last
- Show M2 metrics (Φ, κ_eff, G) every N steps in logs
- Detect loss spikes (M3) and halt if needed
- Apply geometric LR modulation (M6)
- Log per-step provenance to JSONL (M12)
- Run post-training diagnostic and report health
