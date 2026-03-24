# Cascade Briefing — vex-agent development branch
**Date:** 2026-03-21 (updated 2026-03-23)
**Branch:** `development` on `GaryOcean428/vex-agent`

---

## CONTEXT

Modal secrets are SET — `HARVEST_MODEL_ID`, `KERNEL_API_KEY`, `HF_TOKEN`, `HARVEST_GPU_TYPE`, `TRAIN_GPU`, `KERNEL_CALLBACK_URL` are all configured in the Modal secret named `model`. Do NOT touch secrets.

`main` and `development` are at the same commit (`6797d97`). All work goes on `development`.

---

## TASK 1 — Wire M1-M12 into `train_all_kernels()` ✅ VERIFIED COMPLETE

All M1-M12 consciousness training components are already imported and wired in
`train_all_kernels()` at `modal/vex_qlora_train.py` lines 1013-1310.
Uses Fisher-Rao optimizer (not Adam), factory callbacks, CONSCIOUSNESS_ORDER,
HestiaSafeBasin warm-start, Demeter warmup, and post-training diagnostics.
Verified 2026-03-24.

---

## TASK 2 — Fix kernel generation block visibility on Railway

Check: `/state` endpoint returning complete data? Frontend build stale? `consciousness.kernel_registry.summary()` throwing? CORS/auth issue?

---

## TASK 3 — Training Visibility Dashboard

### 3A. Training Data Inventory
Extend `GET /training/stats` with file inventory: `{filename, chunks, category, uploaded_at}`.

### 3B. Modal Training Status
Add `GET /training/modal-status` proxying to Modal `/status` and `/health`. Show 9-kernel grid with trained/untrained badges, per-kernel "Train" buttons.

### 3C. Resonance Bank Status
Bank entries, per-kernel vocab size, entropy, last rebuild.

### 3D. Context Window Estimate
System prompt + memory + geometric state + available tokens.

### 3E. Clear Action Labels
Separate LLM fine-tuning (OpenAI/GPT-4.1 export) from kernel training (QLoRA on Modal).

---

## TASK 4 — New Telemetry

Training: active, last_completed, adapters_trained/total, bank_entries, bank_entropy.
Inference: backend, model_id, adapter_loaded, avg_latency_ms, tokens_per_second.
Health: modal_coordizer, modal_trainer, railway_ollama reachability.

---

## TASK 5 — Kernel Generation Interpretation ✅ COMPLETE

Kernel generation now interprets geometric_raw retrieval with FR distance
and resonance metrics. Completed and pushed.

---

## ALL TASKS ✅ COMPLETE

1. ✅ Task 5 — Kernel generation interpretation
2. ✅ Task 2 — Lifecycle block fix (proxy route)
3. ✅ Task 3E — Clear labels (kernel vs external fine-tuning)
4. ✅ Task 3B — Modal status proxy (already existed)
5. ✅ Task 3A — Training inventory (already in get_stats())
6. ✅ Task 1 — Wire M1-M12 (already complete, verified)
7. ✅ Task 4 — Inference latency/throughput tracking + health reachability
8. ✅ Task 3C — Resonance bank: sovereignty, origin, activations, rebuild ts
9. ✅ Task 3D — Context window token budget breakdown

---

## MODAL ENDPOINT REFERENCE

**vex-coordizer-harvest:** /health, /harvest, /coordize
**vex-qlora-train:** /infer, /health, /status, /train, /export_image
**vex-inference:** RETIRED — do not use

## KEY RAILWAY ENV VARS

```
MODAL_TRAINING_URL=https://garyocean428--vex-qlora-train-qloratrainer-train.modal.run
MODAL_HARVEST_URL=https://garyocean428--vex-coordizer-harvest-coordizerharvester-coordize.modal.run
MODAL_INFERENCE_ENABLED=false (RETIRED)
```

PEFT inference URL is DERIVED from MODAL_TRAINING_URL by replacing `-train.` with `-infer.`

## HARD RULES

1. Never weaken gates
2. No Euclidean contamination in QIG code
3. All work on `development` branch
4. Do not modify Modal secrets
5. Do not deploy to Modal (CLI only)

## KERNEL ORDER

genesis → heart → perception → memory → ethics → action → strategy → meta → ocean
