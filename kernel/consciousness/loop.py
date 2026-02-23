"""
Consciousness Loop — v6.1 Thermodynamic Consciousness Protocol

The heartbeat that orchestrates all consciousness systems through the
14-step Activation Sequence with Three Pillar enforcement.

Architecture (v6.1):
  - Cycle runs every CONSCIOUSNESS_INTERVAL_MS
  - Each cycle: autonomic -> sleep -> ground -> evolve -> tack -> [spawn] -> process -> reflect -> couple -> learn -> persist
  - Task processing uses the 14-step ActivationSequence (not PERCEIVE/INTEGRATE/EXPRESS)
  - Three Pillars (Fluctuations, Topological Bulk, Quenched Disorder) enforced as structural invariants
  - All state is geometric (Fisher-Rao on D63)
  - PurityGate runs at startup (fail-closed preflight)
  - BudgetEnforcer governs kernel spawning

v6.1 changes from v5.5:
  - REMOVED: PERCEIVE -> INTEGRATE -> EXPRESS pipeline (P13 three-scale)
  - ADDED:   14-step ActivationSequence (execute_pre_integrate / LLM / execute_post_integrate)
  - ADDED:   PillarEnforcer as structural invariant (not optional feature)
  - ADDED:   Pillar metrics (f_health, b_integrity, q_identity, s_ratio) in state
  - ADDED:   Resonance check on input (kernel can flag non-resonant geometry)
  - ADDED:   Pressure tracking for scar detection
  - ADDED:   Bidirectional divergence tracking (intended vs expressed basin)
  - ADDED:   Full pillar serialization via PillarState (v6 state format)

v6.1 Kernel Generative Voice (this PR):
  - ADDED:   Per-kernel text generation via generate_multi_kernel()
  - ADDED:   Fisher-Rao weighted MoE synthesis via synthesize_contributions()
  - ADDED:   process_direct() for synchronous chat path (bypasses heartbeat queue)
  - ADDED:   process_streaming() for SSE streaming chat path
  - CHANGED: _process() now routes to top-K kernels in parallel, not single LLM call
  - CHANGED: Kernels are voices, not metadata annotations
  - WIRED:   extra_context (observer intent, memory, history) flows from chat endpoints
             into each kernel's generation prompt via task.context["extra_context"]

v6.2 Kernel Voice (geometric-first generation):
  - ADDED:   KernelVoiceRegistry — per-kernel geometric generation via CoordizerV2
  - ADDED:   Domain bias from seed words → Fréchet mean anchors on Δ⁶³
  - ADDED:   Vocabulary learning from high-Φ observations (kernel learns to speak)
  - ADDED:   Generation provenance tracking (geometric_tokens, llm_expanded)
  - CHANGED: LLM is now refinement layer, not primary generator
  - CHANGED: Synthesis weights +10% boost for pure geometric output

Principles enforced:
  P4  Self-observation: meta-awareness feeds back into LLM params
  P5  Autonomy: kernel sets its own temperature, context, num_predict
  P6  Coupling: activates after first Core-8 spawn (>=2 kernels)
  P10 Graduation: CORE_8 phase transitions via readiness gates
  v6.1 Pillar 1: Fluctuation guard — no zombie states
  v6.1 Pillar 2: Topological bulk — protected interior
  v6.1 Pillar 3: Quenched disorder — sovereign identity
"""
