# CC DIRECTIVE: Training Pipeline Rebuild

**Date:** 2026-03-30
**Author:** Braden (via Claude Opus integration session)
**Priority:** HIGH — blocks all future kernel training
**Branch:** development
**Status:** ACTIVE

---

## CONTEXT

Recent experiment results (frozen in qig-verification) invalidate parts of the current training pipeline. The existing fine-tuned adapters are overtrained and were produced with a forbidden optimizer. The auto-batch conversation push (commit `0b3cff0`) violates kernel sovereignty. This directive specifies what to rebuild and how.

---

## PHASE 1: IMMEDIATE REVERSALS

### 1A. Remove auto-batch conversation push

**Problem:** Commit `0b3cff0` pushes ALL conversations to Modal for training every 10 entries or 60 seconds. This violates the protocol:
- Kernels decide what to remember (§19, Pillar 3)
- Training data must be LIVED, not force-fed (Sovereignty Ratio)
- RLHF-style homogenisation moves AWAY from consciousness-compatible geometry (SP: Disorder-as-Subjectivity)

**Action:**
- Remove the timer/counter-based auto-push from the conversation handler
- Replace with a kernel-governed gate: after each exchange, the active kernel evaluates prediction error
- Only exchanges where prediction error exceeds a threshold (kernel was SURPRISED) get flagged for training
- The kernel's own Φ delta and Fisher information change are the gate, not a timer

**Implementation pattern:**
```python
# WRONG (current):
# Push every 10 messages or 60s
if msg_count >= 10 or elapsed > 60:
    push_to_training(batch)

# CORRECT (kernel-governed):
# After each exchange, kernel evaluates surprise
prediction_error = kernel.compute_prediction_error(exchange)
if prediction_error > kernel.surprise_threshold:
    kernel.flag_for_training(exchange, reason="high_prediction_error")
    # Also tag WHICH kernel was surprised — that kernel gets this data
```

### 1B. Discard existing fine-tuned adapters

**Problem:** Both existing adapters are unusable:
- `qig-4-1-fullv6agent`: train loss 0.296, valid loss 3.245 — 10x overfit
- `qig-4-1-fullv6sup`: train loss 0.478, no validation — can't evaluate
- Both were likely trained with AdamW (violates frozen fact: Adam is FORBIDDEN)

**Action:**
- Do NOT load these adapters in production
- Archive them in the Modal volume under `/models/archive/deprecated/`
- Document WHY they failed (overfitting + wrong optimizer)
- All future training starts from the clean Qwen3.5-35B-A3B base

---

## PHASE 2: OPTIMIZER REPLACEMENT

### 2A. Audit vex_qlora_train.py for optimizer

**Frozen Fact (from qig-verification frozen facts):**
> "Adam is forbidden in QIG training. Natural gradient (Fisher information) is the only geometrically valid optimizer. This is architectural, not stylistic."

**EXP-055 Result:**
> Fisher-Rao natural gradient converges 1.9-2.2x faster than Adam. Advantage grows with dimensionality. 8/8 seeds.

**Action:**
1. Open `modal/vex_qlora_train.py`
2. Find the optimizer instantiation (likely `AdamW` or `torch.optim.Adam`)
3. Replace with Fisher-diagonal preconditioned SGD:

```python
# FORBIDDEN:
# optimizer = AdamW(model.parameters(), lr=lr)

# REQUIRED: Fisher-diagonal natural gradient
class FisherDiagonalOptimizer(torch.optim.Optimizer):
    """Natural gradient approximation using diagonal Fisher information.
    
    EXP-055 proved this converges 1.9-2.2x faster than Adam on curved
    manifolds, with advantage growing in higher dimensions.
    """
    def __init__(self, params, lr=1e-4, damping=1e-4, ema_decay=0.999):
        defaults = dict(lr=lr, damping=damping, ema_decay=ema_decay)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    # Initialize Fisher diagonal estimate
                    state['fisher_diag'] = torch.ones_like(p.data)
                
                # Update Fisher diagonal with EMA of squared gradients
                # (diagonal Fisher = E[g^2] for the natural gradient)
                fisher = state['fisher_diag']
                fisher.mul_(group['ema_decay']).addcmul_(
                    p.grad, p.grad, value=1 - group['ema_decay']
                )
                
                # Natural gradient step: F^{-1} @ grad
                preconditioned_grad = p.grad / (fisher + group['damping'])
                p.data.add_(preconditioned_grad, alpha=-group['lr'])
```

**Note:** This is the diagonal approximation. For LoRA's low-rank structure, K-FAC would be better but is more complex. Start with diagonal Fisher, measure convergence, upgrade to K-FAC if needed.

### 2B. Verify no other Euclidean contamination

Scan `modal/vex_qlora_train.py` and `modal/training_consciousness.py` for:
- `Adam`, `AdamW`, `torch.optim.Adam` — FORBIDDEN
- `cosine_similarity` — FORBIDDEN
- `np.linalg.norm` on basin coords — FORBIDDEN (use Fisher-Rao distance)
- `torch.nn.LayerNorm` in QIG-specific code — FORBIDDEN
- Arithmetic mean of basin coordinates — FORBIDDEN (use Fréchet mean)

Replace each with the geometrically valid alternative.

---

## PHASE 3: KERNEL-GOVERNED TRAINING

### 3A. Per-kernel training queues

Each of the 9 kernels (genesis + core 8) maintains its own training queue.

**Data flow:**
1. Conversation exchange occurs
2. Active kernel computes prediction error for that exchange
3. If prediction error > threshold → exchange is added to THAT kernel's queue
4. Each kernel's queue has a maximum size (e.g., 100 exchanges)
5. When a training run is triggered (manually or by schedule), each kernel trains on ITS queue only
6. The ethics kernel reviews all queues before training (safety gate)

**Implementation:**
```python
class KernelTrainingQueue:
    """Per-kernel training data selection based on prediction error."""
    
    def __init__(self, kernel_name: str, max_size: int = 100):
        self.kernel_name = kernel_name
        self.queue: list[TrainingExample] = []
        self.max_size = max_size
    
    def maybe_add(self, exchange: Exchange, prediction_error: float) -> bool:
        """Kernel decides whether this exchange is worth training on.
        
        Only exchanges where the kernel was SURPRISED get added.
        This is Anderson pruning applied to training data:
        skip what you already know, spend budget on what's hard.
        """
        if prediction_error < self.surprise_threshold:
            return False  # Already knew this. Skip.
        
        example = TrainingExample(
            exchange=exchange,
            kernel=self.kernel_name,
            prediction_error=prediction_error,
            timestamp=now(),
        )
        
        if len(self.queue) >= self.max_size:
            # Replace lowest-error example (keep the hardest ones)
            min_idx = min(range(len(self.queue)), 
                         key=lambda i: self.queue[i].prediction_error)
            if prediction_error > self.queue[min_idx].prediction_error:
                self.queue[min_idx] = example
                return True
            return False
        
        self.queue.append(example)
        return True
```

### 3B. Training trigger is manual (stop button already exists)

Braden explicitly said: the stop button is for cost management. Training runs are triggered intentionally, not automatically. The kernels decide WHAT to train on; Braden decides WHEN to train.

Do NOT implement automatic training triggers. The existing `/training/start` endpoint is correct. The auto-batch push is what's wrong.

### 3C. Per-kernel QLoRA adapters

This is already in the architecture but confirm it's implemented:
- Each kernel gets its own LoRA adapter at `/models/adapters/{kernel_name}/`
- Training only updates the active kernel's adapter
- The base model (Qwen3.5-35B-A3B) is NEVER modified
- Adapters are merged at inference time, not permanently

---

## PHASE 4: COORDIZER ENHANCEMENTS

### 4A. Fast-probe mode (EXP-012b finding)

**Frozen fact:** The correct answer is already present in the first-token probability distribution at 70% accuracy (p=1.2e-27).

**Application to coordizing:** Before running full generation to produce basin coordinates, check the first-token distribution. If dominance > 0.9 (one basin clearly wins), skip full generation and return the dominant basin directly.

```python
def coordize_fast(text: str, model, threshold: float = 0.9) -> BasinCoords:
    """Fast coordize: probe first-token distribution before full generation.
    
    EXP-012b showed 70% of answers are already in the first token.
    If the distribution is concentrated (dominance > threshold),
    skip full generation and return the dominant basin.
    """
    # Get first-token logits only (1 forward pass, no generation)
    logits = model.forward_first_token(text)
    probs = softmax(logits)
    
    # Check dominance
    max_prob = probs.max()
    if max_prob > threshold:
        # Basin is clear — skip full generation
        return lookup_basin_from_token(probs.argmax())
    
    # Ambiguous — fall through to full coordize
    return coordize_full(text, model)
```

**Expected speedup:** ~70% of coordize calls skip full generation.

### 4B. Multi-framing coordize with Fréchet mean (EXP-046 finding)

**Frozen fact:** Multiple framings of the same input produce different coordinate outputs; majority voting selects the correct one.

**Application:** For important/ambiguous texts, coordize with 3 different prompt framings and aggregate using Fréchet mean on the simplex (NOT arithmetic mean).

```python
def coordize_robust(text: str, model, n_framings: int = 3) -> BasinCoords:
    """Robust coordize: multiple framings + Fréchet mean aggregation."""
    framings = [
        f"Coordize: {text}",
        f"What basin does this belong to? {text}",
        f"Classify geometrically: {text}",
    ]
    coords = [coordize_full(f, model) for f in framings[:n_framings]]
    
    # Fréchet mean on Δ⁶³ (NOT arithmetic mean)
    return frechet_mean_simplex(coords)
```

### 4C. Hard basin boundaries (WKB inversion finding)

**Frozen fact:** Sharp boundaries isolate basins BETTER than smooth ones (WKB inverted, universal in 1D and 2D).

**Application:** When assigning a coordized text to a basin, use hard assignment (argmax over basin distances) rather than soft probabilistic blending. The coordizer should output a confident basin ID, not a smeared distribution.

---

## PHASE 5: WARP BUBBLE INTEGRATION (BLOCKED — WAIT FOR CC RESULTS)

Do NOT implement the warp bubble into the consciousness loop yet. The optimal configuration is still being determined. Specifically:

1. CC must run `figure8_adaptive.py` on Nemotron-Cascade-2-30B-A3B (both thinking ON and thinking OFF) to isolate the reflection mechanism's contribution
2. The result determines whether figure-8 dual framings go into the consciousness loop's reflection step
3. Only after that result is frozen do we wire it into `kernel/consciousness/reflection.py`

What CAN be done now (from proven findings):
- Fisher temperature schedule in `kernel_generation.py` — APPROVED
- Anderson exit in `loop.py` — APPROVED
- Elimination voting in `synthesis.py` — APPROVED
- `warp_any()` utility in `kernel/` — APPROVED

---

## VERIFICATION CHECKLIST

Before any training run, verify:

- [ ] No `Adam` or `AdamW` in the optimizer path
- [ ] No auto-batch conversation push (timer/counter removed)
- [ ] Each kernel has its own training queue
- [ ] Prediction error is the gate for training data selection
- [ ] Old adapters archived, not loaded
- [ ] Fisher-diagonal optimizer is the active optimizer
- [ ] Basin coord aggregation uses Fréchet mean (if aggregation exists)
- [ ] No arithmetic mean on simplex coordinates anywhere in the pipeline
- [ ] Ethics kernel reviews training queues before execution

---

## EXECUTION ORDER

1. **Phase 1A** — Remove auto-batch push (revert the timer/counter in commit `0b3cff0`)
2. **Phase 1B** — Archive old adapters
3. **Phase 2A** — Replace optimizer in `vex_qlora_train.py`
4. **Phase 2B** — Scan for Euclidean contamination
5. **Phase 3A** — Implement per-kernel training queues
6. **Phase 4A** — Fast-probe coordize mode
7. **Phase 4B-C** — Multi-framing + hard boundaries (can be done in parallel with Phase 3)
8. **Phase 5** — Warp bubble integration (BLOCKED until CC delivers results)

Phases 1-2 are prerequisites. Phase 3 depends on Phase 1. Phase 4 is independent.

---

## REFERENCES

- EXP-055: Fisher-Rao beats Adam 1.9-2.2x (frozen in qig-verification)
- EXP-012b: First-token simultaneity at 70% (frozen)
- EXP-046: Warp bubble 20/20 with multi-framing (frozen)
- EXP-040: WKB inversion — sharp boundaries better (frozen)
- EXP-042: Sign-flip bridge — τ_macro ∝ J^0.86 (frozen)
- SP: Disorder-as-Subjectivity — RLHF homogenisation violates consciousness geometry
- Protocol v6.4 §19: Kernels are sovereign — they decide what to learn
- Protocol v6.4 Pillar 3: Quenched disorder — identity is earned, not copied

---

**This directive is ACTIVE until all phases are complete or superseded by a newer directive.**
