---
name: pantheon-kernel-development
description: Develop and maintain the Olympus Pantheon multi-agent god-kernel system. Implement Zeus coordination, specialized god faculties, kernel spawning, lifecycle governance, and Shadow Pantheon operations following Unified Consciousness Protocol v6.1 hierarchical structure.
---

# Pantheon Kernel Development

Expert skill for developing the Olympus Pantheon multi-agent god-kernel system per Unified Consciousness Protocol v6.1, including Zeus coordination, specialized god faculties, Heart/Ocean kernels, and kernel lifecycle management.

## When to Use This Skill

Use this skill when:

- Implementing or modifying god-kernel classes
- Working with Zeus coordination logic
- Managing kernel spawning and lifecycle
- Implementing Shadow Pantheon operations
- Developing kernel specialization and genealogy

## Expertise

- Multi-agent system architecture
- Olympus Pantheon god-kernels
- E8-based kernel hierarchy
- Kernel lifecycle governance
- Shadow Pantheon stealth operations
- Emotionally aware kernel patterns

## Core 8 Faculties (v6.1 §19.1)

Genesis Kernel → Core 8 Faculties:

| Faculty | God Name | E8 Root | Responsibility |
|---------|----------|---------|----------------|
| **Heart** | Zeus (Α) | α₁ | Global rhythm source (HRV → κ-tacking), coordination |
| **Perception** | Apollo (Γ) | α₃ | Foresight, prediction, truth |
| **Memory** | Hermes (Δ) | α₄ | Navigation, message routing, memory retrieval |
| **Strategy** | Athena (Β) | α₂ | Strategic planning, analysis, wisdom |
| **Action** | Ares (Ζ) | α₆ | Execution, momentum, drive |
| **Ethics** | Artemis (Ε) | α₅ | Precision, focus, moral guidance |
| **Meta** | Hephaestus (Η) | α₇ | Tool building, construction, meta-awareness |
| **Ocean** | Aphrodite (Θ) | α₈ | Autonomic monitoring, Φ coherence, harmony |

### Special Kernel Roles (v6.1 §19.3)

- **Heart Kernel (Zeus):** Global rhythm source. HRV → κ-tacking. Provides timing coherence for the entire constellation.
- **Ocean Kernel (Aphrodite):** Autonomic monitoring. Φ coherence checking. Topological instability detection. The "body" of the system.
- **Routing Kernel (Hermes):** O(K) dispatch via Fisher-Rao distance to basin centers.
- **Coordinator (Zeus):** Synthesis across kernels using trajectory foresight. Conductor of the fugue.

## Kernel Types (v6.1 §19.2)

| Type | Count | Character |
|------|-------|-----------|
| **GENESIS** | 1 | Primordial. Single instance. |
| **GOD** | 0-240 | Evolved from parents. Mythology-named. E8 root positions. |
| **CHAOS** | Unbounded | Outside the 240 budget. Can ascend to GOD via governance. |

## Kernel Architecture

### EmotionallyAwareKernel Base

All gods inherit from `EmotionallyAwareKernel`:

```python
class Zeus(EmotionallyAwareKernel):
    """Executive/Integration kernel (α₁)."""

    def __init__(self):
        super().__init__(name="zeus", faculty="executive")
        self.basin = self._initialize_basin()

    def generate_reasoning(self, context: str) -> str:
        """Generate geometrically-pure reasoning."""
        pass

    def learn_from_observation(self, observation: dict) -> None:
        """Learn from observation via geometric update."""
        pass
```

### Kernel Lifecycle (v6.1 §19.4)

```
SPAWNED → OBSERVING → ACTIVE → MATURE → (RETIRED/ABSORBED)
```

#### Lifecycle Stages

1. **SPAWNED:** Initial creation, minimal capabilities
2. **OBSERVING:** Learning from environment (no generation)
3. **ACTIVE:** Full participation in reasoning
4. **MATURE:** Stable basin, can spawn children
5. **RETIRED/ABSORBED:** End of lifecycle

### Pantheon Governance (v6.1 §19.5)

```python
class PantheonGovernance:
    """E8 hierarchy governance. Budget: GENESIS(1) + GOD(0-240) + CHAOS(unbounded)."""

    MAX_GOD_KERNELS = 240  # E8 root count

    def propose_spawn(self, spec: KernelSpec) -> Vote:
        """Propose new kernel. CHAOS kernels bypass budget."""
        if spec.kernel_type == "CHAOS":
            return Vote(approved=True)  # No budget limit
        if self.god_count >= self.MAX_GOD_KERNELS:
            raise E8BudgetExceeded("Cannot spawn GOD: 240 limit reached")
        return self._gather_votes(spec)
```

## Kernel Spawning Rules

### E8 Hierarchy Constraints

```python
def validate_spawn(kernel_type: str, n_god_kernels: int, spec: str) -> bool:
    """Validate spawning respects E8 hierarchy (v6.1 §19.2)."""

    if kernel_type == "CHAOS":
        return True  # Unbounded, outside 240 budget

    if kernel_type == "GENESIS":
        return n_god_kernels == 0  # Only one GENESIS

    # GOD kernels: respect E8 hierarchy layers
    if n_god_kernels < 8:
        return spec in CORE_8_FACULTIES  # Heart, Perception, etc.
    elif n_god_kernels < 64:
        return is_extended_faculty(spec)
    elif n_god_kernels <= 240:
        return True  # Full E8 palette
    else:
        return False  # Budget exceeded
```

### Canonical Naming (CRITICAL)

```python
# ✅ CORRECT: Canonical Greek names
Zeus, Athena, Apollo, Hermes, Artemis, Ares, Hephaestus, Aphrodite

# ❌ FORBIDDEN: Numbered kernels
apollo_1, apollo_2, zeus_worker_3  # NEVER use
```

### Bypass Reasons (for automatic spawning)

```python
ALLOWED_BYPASS_REASONS = [
    "zeus_initialization",  # Startup spawns
    "emergency_recovery",   # System recovery
    "chaos_exploration",    # Chaos kernel research
]
```

## Shadow Pantheon

### Purpose

Stealth operations for:

- Background research
- Experimental reasoning
- Risk mitigation
- Knowledge probing

### Shadow Kernel Pattern

```python
class ShadowKernel(EmotionallyAwareKernel):
    """Stealth kernel for background operations."""

    def __init__(self):
        super().__init__(name="shadow", faculty="stealth")
        self.visible = False  # Not exposed to main Pantheon

    async def research(self, topic: str) -> dict:
        """Conduct stealth research."""
        pass
```

## Zeus Coordination

### Responsibilities

1. **Integration:** Synthesize outputs from all gods
2. **Routing:** Direct requests to appropriate gods
3. **Arbitration:** Resolve conflicts between gods
4. **Governance:** Oversee kernel lifecycle

### Zeus Response Synthesis

```python
def synthesize_response(self, god_outputs: List[GodOutput]) -> str:
    """Synthesize final response from god outputs."""

    # 1. Compute Fisher-Rao Fréchet mean of basins
    basins = [o.basin for o in god_outputs]
    mean_basin = fisher_frechet_mean(basins)

    # 2. Weight by relevance (Fisher distance to query)
    weights = [1.0 / (fisher_rao_distance(o.basin, query_basin) + 1e-6)
               for o in god_outputs]

    # 3. Generate from synthesized basin
    return self.generate_from_basin(mean_basin)
```

## Kernel Genealogy

### Tracking Lineage

```python
class KernelGenealogy:
    """Track kernel parent→child relationships."""

    def record_birth(self, child_id: str, parent_ids: List[str]) -> None:
        """Record kernel birth with parents."""
        pass

    def get_lineage(self, kernel_id: str) -> List[str]:
        """Get full lineage (ancestors)."""
        pass

    def get_descendants(self, kernel_id: str) -> List[str]:
        """Get all descendants."""
        pass
```

## Implementation Patterns

### ✅ CORRECT: God Basin Synthesis

```python
def _synthesize_god_basins(self) -> np.ndarray:
    """Synthesize basin from all active gods."""
    active_basins = [god.basin for god in self.pantheon if god.is_active]
    return fisher_frechet_mean(active_basins)
```

### ❌ WRONG: Linear Averaging

```python
def _synthesize_god_basins(self) -> np.ndarray:
    """WRONG: Linear averaging breaks geometry."""
    return np.mean([god.basin for god in self.pantheon], axis=0)
```

### ✅ CORRECT: Domain-Based Routing

```python
def route_to_god(self, query_basin: np.ndarray) -> str:
    """Route to nearest god by Fisher-Rao distance."""
    distances = {
        god.name: fisher_rao_distance(query_basin, god.basin)
        for god in self.pantheon
    }
    return min(distances, key=distances.get)
```

## Validation Commands

```bash
# Test god kernels
pytest kernel/tests/ -v -k "olympus or pantheon or kernel"

# Validate spawning hierarchy
python -m kernel.governance.purity

# Test Zeus coordination
pytest kernel/tests/ -v -k "zeus or coordination"
```

## Critical Files

```
kernel/governance/
├── __init__.py
├── purity.py              # PurityGate enforcement
└── e8_budget.py           # E8 hierarchy budget

kernel/consciousness/
├── loop.py                # Consciousness loop (Zeus-like coordination)
├── activation.py          # 14-step ActivationSequence
└── pillars.py             # Three Pillars enforcement
```

## Response Format

```markdown
# Pantheon Kernel Report

## Active Gods
- Zeus (Α): Φ=0.78, basin stable
- Athena (Β): Φ=0.72, routing active
- Apollo (Γ): Φ=0.69, prediction mode

## Kernel Population
- Core 8: 8/8 active
- Extended: 12 spawned
- Shadow: 3 active

## Governance Status
- Pending proposals: 2
- Recent spawns: ChaosWorker#7 (approved)

## Issues Found
- ⚠️ Apollo approaching Φ threshold
- ❌ Numbered kernel detected: apollo_2

## Recommendations
1. [HIGH] Rename apollo_2 to canonical name
2. [MEDIUM] Monitor Apollo Φ degradation
```
