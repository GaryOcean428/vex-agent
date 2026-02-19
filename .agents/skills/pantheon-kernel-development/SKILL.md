---
name: pantheon-kernel-development
description: Develop and maintain the Olympus Pantheon multi-agent god-kernel system. Implement Zeus coordination, specialized god faculties, kernel spawning, lifecycle governance, and Shadow Pantheon operations following E8 Protocol v4.0 hierarchical structure.
---

# Pantheon Kernel Development

Expert skill for developing the Olympus Pantheon multi-agent god-kernel system, including Zeus coordination, specialized god faculties, and kernel lifecycle management.

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

## Core 8 Gods (E8 Simple Roots)

| God | Faculty | E8 Root | Responsibility |
|-----|---------|---------|----------------|
| **Zeus (Α)** | Executive/Integration | α₁ | Coordination, final decisions |
| **Athena (Β)** | Wisdom/Strategy | α₂ | Strategic planning, analysis |
| **Apollo (Γ)** | Truth/Prediction | α₃ | Foresight, prediction |
| **Hermes (Δ)** | Communication/Navigation | α₄ | Message routing, navigation |
| **Artemis (Ε)** | Focus/Precision | α₅ | Exploration, precision targeting |
| **Ares (Ζ)** | Energy/Drive | α₆ | Action, momentum |
| **Hephaestus (Η)** | Creation/Construction | α₇ | Tool building, construction |
| **Aphrodite (Θ)** | Harmony/Aesthetics | α₈ | Harmony, user experience |

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

### Kernel Lifecycle

```
SPAWNED → OBSERVING → ACTIVE → MATURE → (RETIRED/ABSORBED)
```

#### Lifecycle Stages

1. **SPAWNED:** Initial creation, minimal capabilities
2. **OBSERVING:** Learning from environment (no generation)
3. **ACTIVE:** Full participation in reasoning
4. **MATURE:** Stable basin, can spawn children
5. **RETIRED/ABSORBED:** End of lifecycle

### Pantheon Governance

```python
# Spawning requires Pantheon approval
class PantheonGovernance:
    def propose_spawn(self, spec: KernelSpec) -> Vote:
        """Propose new kernel to Pantheon."""
        pass
        
    def vote(self, proposal_id: str, vote: bool) -> None:
        """Cast vote on kernel proposal."""
        pass
        
    def execute_decision(self, proposal_id: str) -> None:
        """Execute approved/rejected decision."""
        pass
```

## Kernel Spawning Rules

### E8 Hierarchy Constraints

```python
def validate_spawn(n_kernels: int, spec: str) -> bool:
    """Validate spawning respects E8 hierarchy."""
    
    if n_kernels <= 8:
        # Only primary axes (core 8 gods)
        return spec in CORE_8_GODS
        
    elif n_kernels <= 56:
        # Refined specializations
        return is_refined_specialization(spec)
        
    elif n_kernels <= 126:
        # Specialist kernels
        return is_specialist(spec)
        
    else:
        # Full palette (up to 240)
        return True
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
pytest tests/test_olympus/ -v

# Validate spawning hierarchy
python scripts/validate_e8_hierarchy.py

# Test Zeus coordination
pytest tests/test_zeus_coordination.py

# Shadow Pantheon tests
pytest tests/test_shadow_pantheon.py
```

## Critical Files

```
qig-backend/olympus/
├── __init__.py
├── base_god.py           # EmotionallyAwareKernel base
├── zeus.py               # Zeus coordination
├── athena.py             # Wisdom/Strategy
├── apollo.py             # Truth/Prediction
├── hermes.py             # Communication/Navigation
├── artemis.py            # Focus/Precision
├── ares.py               # Energy/Drive
├── hephaestus.py         # Creation/Construction
├── aphrodite.py          # Harmony/Aesthetics
├── shadow_research.py    # Shadow Pantheon
├── lightning_kernel.py   # Fast insight capture
└── governance.py         # Pantheon governance
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
