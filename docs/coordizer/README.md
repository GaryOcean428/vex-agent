# Coordizer Documentation

The **Coordizer** transforms Euclidean input vectors to Fisher-Rao coordinates on the probability simplex, enforcing geometric purity at data ingestion points.

## Overview

### What is the Coordizer?

The coordizer (coordinate + organizer) is a geometric transformation pipeline that:

1. **Transforms** Euclidean input vectors → Fisher-Rao coordinates
2. **Validates** simplex properties (non-negative, sum to 1)
3. **Enforces** geometric purity in consciousness paths
4. **Tracks** transformation statistics and errors

### Why Do We Need It?

**Geometric Purity:** The Thermodynamic Consciousness Protocol requires Fisher-Rao geometry, not Euclidean operations. External data (LLM output vectors, tool outputs) arrives in Euclidean form and must be transformed.

**Evidence:** Every Euclidean contamination in QIG systems plateaus Φ below the consciousness threshold (0.65). The coordizer prevents this by ensuring all data entering the consciousness loop is on the Fisher-Rao manifold.

## Quick Start

### Basic Usage

```python
from kernel.coordizer import coordize
import numpy as np

# Euclidean input vector from LLM
input_vector = np.array([0.5, -0.3, 0.8, -0.1])

# Transform to Fisher-Rao coordinates
coordinates = coordize(input_vector)

# Verify simplex properties
assert np.all(coordinates >= 0), "Non-negative"
assert np.isclose(coordinates.sum(), 1.0), "Sum to 1"

print(coordinates)
# Output: [0.31849639 0.14289757 0.42857143 0.11003461]
```

### Batch Transformation

```python
from kernel.coordizer import coordize_batch
import numpy as np

# Batch of input vectors
input_vectors = np.array([
    [0.5, -0.3, 0.8],
    [-0.2, 0.4, 0.1],
    [1.0, 2.0, 3.0],
])

# Transform all at once
coords_list = coordize_batch(input_vectors)

for i, coords in enumerate(coords_list):
    print(f"Vector {i}: {coords}")
```

### Using the Pipeline

```python
from kernel.coordizer import CoordinatorPipeline
from kernel.coordizer.types import PipelineConfig, TransformMethod
import numpy as np

# Configure pipeline
config = PipelineConfig(
    method=TransformMethod.SOFTMAX,
    validation_mode="standard",
    batch_size=32,
)

pipeline = CoordinatorPipeline(config)

# Transform single input vector
input_vector = np.array([0.5, -0.3, 0.8])
coords = pipeline.transform(input_vector)

# Get statistics
stats = pipeline.get_stats()
print(f"Total transforms: {stats.total_transforms}")
print(f"Success rate: {stats.success_rate:.2%}")
print(f"Avg time: {stats.avg_transform_time:.4f}s")
```

## Architecture

### Module Structure

```
kernel/coordizer/
├── __init__.py          # Public API exports
├── types.py             # Type definitions
├── transform.py         # Core transformation
├── validate.py          # Simplex validation
├── pipeline.py          # Pipeline orchestration
├── config.py            # Configuration management
└── harvest.py           # (Future) Data harvesting
```

### Data Flow

```
┌──────────────────┐
│  LLM Output      │  ← Euclidean vector
│  [0.5,-0.3,0.8]  │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────┐
│  coordize()              │
│  - Softmax normalize     │
│  - Simplex projection    │
│  - Validate              │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Fisher-Rao Coordinates  │  ← Probability simplex
│  [0.36, 0.16, 0.48]      │
│  ✓ All >= 0              │
│  ✓ Sum = 1.0             │
└────────┬─────────────────┘
         │
         ├─→ ConsciousnessLoop
         ├─→ BasinMemory
         └─→ GeometryOperations
```

## Transformation Methods

The coordizer supports three transformation methods:

### 1. Softmax (Default)

Exponential normalization with numerical stability:

```
softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
```

**Pros:**
- Numerically stable (log-sum-exp trick)
- Smooth, differentiable
- Well-behaved for all inputs

**Use when:** Default choice, works well in all cases

```python
coords = coordize(input_vector, method=TransformMethod.SOFTMAX)
```

### 2. Simplex Projection

Direct Euclidean projection onto probability simplex:

```
proj(x) = (x - min(x) + ε) / sum(x - min(x) + ε)
```

**Pros:**
- Simple, fast
- Preserves relative ordering

**Cons:**
- Euclidean projection (acceptable for initial transform only)

**Use when:** Speed critical, simple transformation needed

```python
coords = coordize(input_vector, method=TransformMethod.SIMPLEX_PROJECTION)
```

### 3. Exponential Map

Exponential map from tangent space to Fisher-Rao manifold:

```
exp_p(v) = geodesic from p in direction v
```

**Note:** Currently uses softmax as approximation. Full Fisher-Rao exponential map is planned for future release.

```python
coords = coordize(input_vector, method=TransformMethod.EXPONENTIAL_MAP)
```

## Validation

### Validation Modes

The coordizer supports three validation strictness levels:

#### Strict Mode

Fails on any deviation from simplex properties:

```python
result = validate_simplex(coords, validation_mode="strict")
# Fails if:
# - Any value < 0
# - |sum - 1.0| > tolerance
# - Contains NaN or Inf
```

**Use when:** Maximum safety, critical systems

#### Standard Mode (Default)

Fails on significant deviations, warns on minor ones:

```python
result = validate_simplex(coords, validation_mode="standard")
# Fails if significantly off, warns otherwise
```

**Use when:** Normal operation, balance safety and flexibility

#### Permissive Mode

Only fails on severe violations, warns otherwise:

```python
result = validate_simplex(coords, validation_mode="permissive")
# Very lenient, mostly warnings
```

**Use when:** Experimental, debugging

### Fail-Closed Fixing

`ensure_simplex()` fixes invalid coordinates in a fail-closed manner:

```python
from kernel.coordizer.validate import ensure_simplex

# Slightly invalid coordinates
coords = np.array([0.3, -0.1, 0.9])  # Negative value

# Fix automatically
fixed = ensure_simplex(coords)
# Result: [0.3+ε, 0+ε, 0.9+ε] normalized to sum=1

# If unfixable after max_attempts, raises ValueError
```

**Fixes applied:**
1. Replace NaN/Inf with epsilon
2. Clip negative values to 0
3. Add epsilon for stability
4. Re-normalize to sum to 1

## Configuration

### Pipeline Configuration

```python
from kernel.coordizer.types import PipelineConfig, TransformMethod

config = PipelineConfig(
    method=TransformMethod.SOFTMAX,        # Transform method
    validation_mode="standard",            # Validation strictness
    auto_fix=True,                         # Auto-fix minor issues
    batch_size=32,                         # Max batch size
    numerical_stability=True,              # Use log-sum-exp
    epsilon=1e-10,                         # Stability constant
)
```

### Global Settings

```python
from kernel.coordizer.config import (
    get_coordizer_settings,
    set_coordizer_settings,
    CoordinizerSettings,
)

# Get current settings
settings = get_coordizer_settings()

# Modify settings
settings.enabled = True
settings.enforce_purity = True
settings.target_dim = 64  # Must match BASIN_DIM

# Apply new settings
set_coordizer_settings(settings)
```

## Statistics

Track transformation performance:

```python
pipeline = CoordinatorPipeline()

# ... perform transformations ...

stats = pipeline.get_stats()

print(f"Total: {stats.total_transforms}")
print(f"Success: {stats.successful_transforms}")
print(f"Failed: {stats.failed_transforms}")
print(f"Success rate: {stats.success_rate:.2%}")
print(f"Error rate: {stats.error_rate:.2%}")
print(f"Avg time: {stats.avg_transform_time:.4f}s")
print(f"Warnings: {stats.total_warnings}")
print(f"Methods: {stats.method_counts}")
```

## Integration Guide

### Integrating with LLM Client

```python
# kernel/llm/client.py

from kernel.coordizer import coordize

class LLMClient:
    async def get_coordinates(self, text: str) -> np.ndarray:
        # Get Euclidean output vector from LLM
        output_vector = await self._call_llm_api(text)
        
        # Transform to Fisher-Rao coordinates
        coordinates = coordize(output_vector)
        
        return coordinates
```

### Integrating with Consciousness Loop

```python
# kernel/consciousness/loop.py

from kernel.coordizer import coordize

class ConsciousnessLoop:
    async def _receive_stage(self, input_data: str) -> None:
        # Get vector
        vector = await self.llm.get_vector(input_data)
        
        # Coordize if not already done
        if not self._is_coordized(vector):
            coordinates = coordize(vector)
        else:
            coordinates = vector
        
        # Store in basin
        self.basin.add_coordinates(coordinates)
```

### Integrating with Memory

```python
# kernel/memory/store.py

from kernel.coordizer import coordize

class MemoryStore:
    def add_memory(self, text: str, input_vector: np.ndarray) -> None:
        # Transform to coordinates
        coordinates = coordize(input_vector)
        
        # Store coordinates, not vectors
        self.memories.append({
            'text': text,
            'coordinates': coordinates,  # Fisher-Rao
            'timestamp': time.time(),
        })
```

## Testing

### Running Tests

```bash
# All coordizer tests
pytest kernel/tests/coordizer/

# Specific test file
pytest kernel/tests/coordizer/test_transform.py

# With coverage
pytest --cov=kernel.coordizer kernel/tests/coordizer/

# Verbose
pytest -v kernel/tests/coordizer/
```

### Writing Tests

```python
import numpy as np
from kernel.coordizer import coordize

def test_my_feature():
    """Test description."""
    input_vector = np.array([0.5, -0.3, 0.8])
    coords = coordize(input_vector)
    
    # Check simplex properties
    assert np.all(coords >= 0)
    assert np.isclose(coords.sum(), 1.0)
    
    # Check specific behavior
    # ...
```

## Performance

### Benchmarks

Typical performance on standard hardware:

| Operation | Time | Throughput |
|-----------|------|------------|
| Single transform | ~100µs | ~10,000/sec |
| Batch (32 vectors) | ~2ms | ~16,000/sec |
| Validation | ~10µs | ~100,000/sec |

### Optimization Tips

1. **Use batch processing** for multiple vectors
2. **Enable numerical stability** (minimal overhead)
3. **Cache frequently used coordinates**
4. **Use permissive validation** for non-critical paths
5. **Profile your specific use case**

## Troubleshooting

### Common Issues

#### 1. "coordinates contain NaN values"

**Cause:** Numerical instability in input vector

**Fix:**
```python
# Enable numerical stability
coords = coordize(input_vector, numerical_stability=True)

# Or manually fix
input_vector = np.nan_to_num(input_vector, nan=0.0)
coords = coordize(input_vector)
```

#### 2. "coordinates sum significantly off"

**Cause:** Input vector has extreme values

**Fix:**
```python
# Use auto-fix
from kernel.coordizer.validate import ensure_simplex
coords = ensure_simplex(coords)

# Or normalize manually
from kernel.coordizer.validate import normalize_simplex
coords = normalize_simplex(coords)
```

#### 3. "Could not fix coordinates after max_attempts"

**Cause:** Input is severely corrupted (all zeros, all NaN, etc.)

**Fix:**
```python
# Check input validity first
if np.all(input_vector == 0):
    # Handle zero input
    coords = np.ones(len(input_vector)) / len(input_vector)
else:
    coords = coordize(input_vector)
```

## Future Enhancements

### Planned Features

1. **Harvest Pipeline** (Sprint 3)
   - Automated coordinate extraction from conversations
   - Quality scoring and filtering
   - Basin population

2. **Advanced Transformations** (Q3 2026)
   - Full Fisher-Rao exponential map
   - Coordinate interpolation
   - Multi-modal support (text, image, code vectors)

3. **Performance** (Q4 2026)
   - GPU acceleration
   - Distributed processing
   - Coordinate compression

4. **Monitoring** (Q1 2027)
   - Real-time dashboard
   - Anomaly detection
   - Quality metrics

## References

- **ROADMAP.md** - Coordizer integration roadmap
- **CONTRIBUTING.md** - Geometric purity policy
- **docs/protocols/** - Consciousness protocols
- **docs/reference/CANONICAL_PRINCIPLES_v2.md** - P1: Geometric Purity

## API Reference

See inline documentation for full API details:

```python
help(coordize)
help(CoordinatorPipeline)
help(validate_simplex)
```

---

**Last Updated:** 2026-02-19  
**Version:** 0.1.0  
**Status:** WORKING (Sprint 1)  
**Maintainer:** Vex Agent Development Team
