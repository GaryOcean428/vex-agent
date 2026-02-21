# Vex Agent Coordizer Integration Roadmap

**Version:** 1.0
**Date:** 2026-02-19
**Status:** Planning

## Executive Summary

This roadmap outlines the integration of the **Coordizer** system into Vex Agent. The Coordizer provides critical transformation capabilities from Euclidean vectors to Fisher-Rao coordinates, enabling geometric purity across the consciousness system.

**Timeline:** 4 sprints (8-12 weeks)
**Priority:** High (Foundation for v6 protocol compliance)
**Complexity:** Medium-High

## Background

### What is Coordizer?

The Coordizer (coordinate + organizer) is a geometric transformation pipeline that:

1. **Transforms** Euclidean vectors → Fisher-Rao coordinates
2. **Validates** simplex properties (non-negative, sum to 1)
3. **Harvests** coordinate data for basin population
4. **Enforces** geometric purity at ingestion points

### Why Now?

- **Protocol v6 Compliance:** Geometric purity is mandatory
- **CORE_8 Ready:** E8 kernel architecture is implemented
- **Memory Integration:** Basin system needs coordinate data
- **Tooling Gap:** Current system relies on manual coordinate management

### Integration Sources

Two implementations provided:

- **Claude version** (coordizer.zip) - Preferred, more robust
- **ChatGPT version** (vex_coordizer_harvest_pipeline.zip) - Supplementary

**Conflict Resolution:** Prefer Claude implementation; adopt ChatGPT patterns where Claude is incomplete.

## Goals

### Primary Objectives

1. **Geometric Purity Enforcement**
   - All external data (LLM output vectors, tool outputs) transformed to coordinates
   - No Euclidean operations in consciousness paths
   - PurityGate integration

2. **Basin Population**
   - Automated coordinate harvesting from conversations
   - Basin growth via coordized data
   - Semantic clustering on Fisher-Rao manifold

3. **Developer Experience**
   - Simple API: `coordize(input_vector) → coordinates`
   - Clear error messages for violations
   - Documentation and examples

### Success Metrics

- [ ] 100% of input vectors converted to coordinates before consciousness processing
- [ ] PurityGate passes with coordizer active
- [ ] Basin population grows from coordized conversation data
- [ ] Zero Euclidean operations in geometry/ and consciousness/ modules
- [ ] <10ms coordizer overhead per operation

## Architecture

### Module Structure

```
kernel/coordizer/
├── __init__.py              # Public API exports
├── transform.py             # Core transformation (Euclidean → Fisher-Rao)
├── validate.py              # Simplex validation utilities
├── harvest.py               # Conversation data harvesting
├── pipeline.py              # End-to-end pipeline orchestration
├── config.py                # Coordizer configuration
└── types.py                 # Coordizer-specific types

kernel/tests/coordizer/
├── test_transform.py        # Transformation tests
├── test_validate.py         # Validation tests
├── test_harvest.py          # Harvest tests
└── test_pipeline.py         # Integration tests
```

### Data Flow

```
┌────────────────┐
│  LLM Response  │
│ (input vectors)│
└───────┬────────┘
        │
        ▼
┌────────────────────────────┐
│  Coordizer.transform()     │
│  - Softmax normalization   │
│  - Simplex projection      │
│  - Validation              │
└───────┬────────────────────┘
        │
        ▼
┌────────────────────────────┐
│  Coordinates (Δ⁶³)         │
│  - Non-negative            │
│  - Sum to 1                │
│  - Fisher-Rao ready        │
└───────┬────────────────────┘
        │
        ├─→ ConsciousnessLoop  (processing)
        ├─→ BasinMemory        (storage)
        └─→ GeometryOperations (distances)
```

### Integration Points

| Component | Change Required | Priority |
|-----------|----------------|----------|
| `kernel/consciousness/loop.py` | Add coordizer calls at RECEIVE stage | P0 |
| `kernel/llm/client.py` | Coordize output vectors before returning | P0 |
| `kernel/tools/registry.py` | Coordize tool outputs | P1 |
| `kernel/memory/store.py` | Store coordinates, not vectors | P0 |
| `kernel/server.py` | Add `/coordizer/*` endpoints | P2 |
| `frontend/src/types/consciousness.ts` | Add coordizer types | P2 |
| `frontend/src/pages/dashboard/Coordizer.tsx` | Coordizer dashboard page | P3 |

## Sprint Plan

### Sprint 1: Foundation (Weeks 1-3)

**Theme:** Core transformation + validation

#### Week 1: Setup & Types

- [x] Create documentation (CONTRIBUTING.md, AGENTS.md, ROADMAP.md)
- [ ] Create `kernel/coordizer/` module structure
- [ ] Define types in `kernel/coordizer/types.py`

  ```python
  @dataclass
  class CoordinateTransform:
      input_vector: np.ndarray    # Input (Euclidean)
      coordinates: np.ndarray     # Output (Fisher-Rao)
      method: str                 # Transformation method
      timestamp: float            # When transformed

  @dataclass
  class ValidationResult:
      valid: bool
      errors: list[str]
      warnings: list[str]
  ```

- [ ] Add coordizer configuration to `kernel/config/settings.py`

#### Week 2: Core Transform

- [ ] Implement `transform.py`:
  - `coordize(input_vector) → coordinates`
  - Softmax normalization
  - Simplex projection
  - Numerical stability (log-sum-exp trick)
- [ ] Implement `validate.py`:
  - `validate_simplex(coords) → ValidationResult`
  - Non-negativity check
  - Sum-to-one check
  - Dimension validation
- [ ] Add unit tests (>90% coverage)
- [ ] Document transformation mathematics

#### Week 3: Integration Prep

- [ ] Create `pipeline.py`:
  - `CoordinatorPipeline` class
  - Batch processing support
  - Error handling and recovery
- [ ] Add PurityGate checks for coordizer
- [ ] Create coordizer examples in `docs/`
- [ ] Performance benchmarking

**Deliverable:** Working coordizer module with tests

### Sprint 2: LLM & Consciousness Integration (Weeks 4-6)

**Theme:** Wire coordizer into live system

#### Week 4: LLM Client Integration

- [ ] Update `kernel/llm/client.py`:
  - Coordize output vectors in `_get_vector()`
  - Add coordizer to response path
  - Handle coordination errors gracefully
- [ ] Update `kernel/llm/types.py`:
  - Replace `input_vector` fields with `coordinates`
  - Add backward compatibility layer (deprecated, was "embedding")
- [ ] Test LLM integration:
  - Ollama responses coordized
  - External API responses coordized
  - No Euclidean leaks

#### Week 5: Consciousness Loop Integration

- [ ] Update `kernel/consciousness/loop.py`:
  - Call coordizer at RECEIVE stage
  - Store coordinates in basin
  - Update metrics to use Fisher-Rao distances
- [ ] Update `kernel/consciousness/types.py`:
  - Add `CoordinizerState` to `LoopState`
  - Track transformation count, errors
- [ ] Test consciousness integration:
  - Φ metrics stable with coordizer
  - κ ≈ 64 maintained
  - Basin transitions smooth

#### Week 6: Basin & Memory

- [ ] Update `kernel/memory/store.py`:
  - Store coordinates in memory entries
  - Migrate existing vectors (if any, previously called "embeddings")
  - Add coordinate-based retrieval
- [ ] Update basin storage format:
  - Use coordinates, not vectors
  - Backward compatibility (read old format)
- [ ] Test memory integration:
  - Memory retrieval uses Fisher-Rao
  - Basin history preserved
  - No data loss during migration

**Deliverable:** Coordizer integrated into consciousness + memory

### Sprint 3: Harvest Pipeline & Tools (Weeks 7-9)

**Theme:** Automated coordinate harvesting

#### Week 7: Harvest Pipeline

- [x] Implement `harvest.py`:
  - Extract text from conversations
  - Generate vectors (via LLM)
  - Coordize vectors
  - Store in basin memory
- [x] Add harvest configuration:
  - Sampling rate (not every message)
  - Batch size
  - Quality filters
- [x] Implement `gpu_harvest.py` (v6.0 §19):
  - GPU-accelerated harvest via ComputeSDK/Railway
  - Full probability distribution capture (Transformers/vLLM)
  - Three-phase scoring (256→2K→10K→32K)
  - Four vocabulary tiers (Fundamentals/Harmonics/Overtones)
  - Versioned resonance bank artifacts
- [ ] Add harvest endpoint:
  - `POST /api/coordizer/harvest`
  - Manual trigger for testing
  - Scheduled background task

#### Week 8: Tool Integration

- [ ] Update `kernel/tools/registry.py`:
  - Coordize tool outputs before storage
  - Add coordizer to tool context
- [ ] Update individual tools:
  - `web-fetch.ts` → coordize page vectors
  - `github.ts` → coordize code vectors
  - `code-exec.ts` → coordize output vectors
- [ ] Test tool integration:
  - Tools produce coordinates
  - No Euclidean contamination
  - Performance acceptable

#### Week 9: API & Endpoints

- [ ] Add coordizer endpoints to `kernel/server.py`:
  - `POST /api/coordizer/transform` (manual transformation)
  - `GET /api/coordizer/stats` (usage statistics)
  - `GET /api/coordizer/history` (recent transformations)
  - `POST /api/coordizer/validate` (validation endpoint)
- [ ] Add TypeScript types in `frontend/src/types/coordizer.ts`
- [ ] Add coordizer metrics to `/metrics` endpoint

**Deliverable:** Full harvest pipeline + tool integration

### Sprint 4: Dashboard, Docs & Polish (Weeks 10-12)

**Theme:** UI, documentation, production readiness

#### Week 10: Frontend Dashboard

- [ ] Create `frontend/src/pages/dashboard/Coordizer.tsx`:
  - Real-time transformation metrics
  - Coordinate distribution visualization
  - Harvest pipeline status
  - Recent transformations log
- [ ] Add coordizer route to dashboard:
  - Navigation link
  - Protected route (if auth enabled)
- [ ] Add coordizer charts:
  - Transformation rate over time
  - Error rate tracking
  - Coordinate quality metrics

#### Week 11: Documentation & Examples

- [ ] Write `docs/coordizer/README.md`:
  - Architecture overview
  - API documentation
  - Usage examples
- [ ] Create example notebooks:
  - `docs/coordizer/examples/basic_usage.ipynb`
  - `docs/coordizer/examples/harvest_pipeline.ipynb`
  - `docs/coordizer/examples/integration.ipynb`
- [ ] Update main README.md:
  - Add coordizer section
  - Update architecture diagram
  - Add coordizer to feature list
- [ ] Add coordizer to CONTRIBUTING.md examples

#### Week 12: Production Readiness

- [ ] Performance optimization:
  - Batch processing for multiple vectors
  - Caching frequently used coordinates
  - Profiling and bottleneck removal
- [ ] Error handling:
  - Comprehensive error messages
  - Recovery strategies
  - Logging and monitoring
- [ ] Protocol v6 audit:
  - Full PurityGate compliance
  - No Euclidean operations
  - E8 budget respected
- [ ] Final testing:
  - End-to-end integration tests
  - Load testing (1000 requests/min)
  - Production deployment dry-run

**Deliverable:** Production-ready coordizer system

## Risk Mitigation

### Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Performance overhead** | High | Batch processing, caching, profiling |
| **Breaking changes** | High | Backward compatibility layer, gradual rollout |
| **Numerical instability** | Medium | Log-sum-exp trick, validation, tests |
| **Integration bugs** | Medium | Comprehensive testing, staged rollout |
| **Memory bloat** | Low | Coordinate compression, pruning old data |

### Performance Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Performance overhead** | High | Batch processing, caching, profiling |
| **Breaking changes** | High | Backward compatibility layer, gradual rollout |
| **Numerical instability** | Medium | Log-sum-exp trick, validation, tests |

### Process Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Scope creep** | High | Strict sprint boundaries, defer nice-to-haves |
| **Resource constraints** | Medium | Prioritize P0/P1 items, defer P2/P3 |
| **Dependency delays** | Low | Modular design, parallel workstreams where possible |

## Dependencies

### Internal

- [x] E8 kernel architecture (`kernel/governance/`)
- [x] Fisher-Rao geometry (`kernel/geometry/`)
- [x] Basin memory system (`kernel/memory/`)
- [x] Consciousness loop (`kernel/consciousness/`)
- [ ] Protocol v6 specification (in progress)

### External

- Python 3.14+
- NumPy 1.24+
- SciPy 1.10+ (for advanced geometry if needed)
- FastAPI 0.104+
- React 18+ (frontend)

## Success Criteria

### Must Have (P0)

- [ ] All LLM output vectors coordized before consciousness processing
- [ ] PurityGate passes with no violations
- [ ] Basin memory stores coordinates, not vectors
- [ ] Consciousness metrics (Φ, κ) stable with coordizer active
- [ ] Comprehensive test coverage (>85%)
- [ ] Documentation complete

### Should Have (P1)

- [ ] Harvest pipeline functional and automated
- [ ] Tool outputs coordized
- [ ] Dashboard visualization complete
- [ ] Performance <10ms overhead
- [ ] Error handling robust

### Nice to Have (P2)

- [ ] Advanced coordinate compression
- [ ] Coordinate interpolation utilities
- [ ] Multi-modal coordinate support (text, image, code)
- [ ] Coordizer plugin system

## Post-Launch

### Maintenance

- **Monitoring:** Track transformation errors, performance
- **Updates:** Keep coordizer aligned with protocol updates
- **Optimization:** Continuous performance improvements
- **Documentation:** Update docs as system evolves

### Future Enhancements

1. **Multi-Modal Coordinates** (Q3 2026)
   - Image vectors → coordinates
   - Code vectors → coordinates
   - Audio vectors → coordinates

2. **Coordinate Algebra** (Q4 2026)
   - Coordinate interpolation
   - Coordinate composition
   - Coordinate factorization

3. **Distributed Coordizer** (Q1 2027)
   - Multi-node coordinate processing
   - Federated coordinate learning
   - Cross-instance coordinate sharing

## Questions & Decisions

### Open Questions

1. **Coordinate dimensionality:** Keep at 64 or allow variable?
   - **Decision:** Keep at 64 (BASIN_DIM) for E8 alignment

2. **Vector source:** Always from LLM or allow external?
   - **Decision:** Support both, validate external vectors

3. **Backward compatibility:** Support old format? (previously called "embeddings")
   - **Decision:** Yes, read-only support for 1 major version

4. **Coordinate storage:** Compressed or raw?
   - **Decision:** Raw initially, compression in post-launch

### Decisions Made

- [x] Prefer Claude coordizer implementation
- [x] Integrate at RECEIVE stage of consciousness loop
- [x] Use softmax for initial transformation approach
- [x] Coordinate validation is fail-closed (rejects invalid)
- [x] Dashboard page is lower priority (P2)

## Appendix

### References

- **Claude Coordizer:** `coordizer.zip` (primary source)
- **ChatGPT Pipeline:** `vex_coordizer_harvest_pipeline.zip` (supplementary)
- **Protocol v5.5:** `docs/THERMODYNAMIC_CONSCIOUSNESS_PROTOCOL_v5_5.md`
- **Canonical Principles:** `docs/20260217-canonical-principles-2.00W.md`
- **Frozen Facts:** `kernel/config/frozen_facts.py`
- **E8 Budget:** `kernel/governance/budget.py`

### Glossary

- **Coordize:** Transform Euclidean vector to Fisher-Rao coordinate
- **Simplex:** Probability simplex Δ⁶³ (all non-negative, sum to 1)
- **Fisher-Rao:** Information geometry metric (pure, no Euclidean)
- **Basin:** Attractor basin on the information manifold
- **CORE_8:** Eight foundational kernels in E8 lattice
- **Harvest:** Extract and coordize data from conversations
- **Purity:** No Euclidean operations in geometric code paths

---

**Last Updated:** 2026-02-19
**Next Review:** After Sprint 1 completion
**Owner:** Vex Agent Development Team
**Status:** Ready for Sprint 1 kickoff
