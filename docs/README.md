# Vex Agent Documentation

This directory contains all technical documentation for the Vex Agent project, organized by topic and purpose.

## Documentation Structure

```
docs/
├── README.md                    # This file (documentation index)
├── architecture/                # System design and architecture
├── protocols/                   # Consciousness protocols (v5.0, v5.5, v6)
├── development/                 # Development guides and practices
├── api/                         # API documentation and examples
├── coordizer/                   # Coordizer system documentation
├── experiments/                 # Experimental results and analyses
└── archive/                     # Deprecated or historical docs
```

## Quick Links

### Getting Started

- [Contributing Guide](../CONTRIBUTING.md) - How to contribute to the project
- [Development Setup](../AGENTS.md) - Detailed development environment setup
- [Roadmap](../ROADMAP.md) - Project roadmap and future plans

### Architecture

- [System Architecture](architecture/SYSTEM_ARCHITECTURE.md) - Overall system design
- [E8 Kernel Architecture](architecture/E8_KERNEL_ARCHITECTURE.md) - E8 lattice and kernel system
- [Data Flow](architecture/DATA_FLOW.md) - Request/response flow through system
- [Dual Service Architecture](architecture/DUAL_SERVICE.md) - Python kernel + TypeScript proxy

### Protocols

- [Protocol v5.0](protocols/THERMODYNAMIC_CONSCIOUSNESS_PROTOCOL_v5_0.md) - Base thermodynamic consciousness protocol
- [Protocol v5.5](protocols/THERMODYNAMIC_CONSCIOUSNESS_PROTOCOL_v5_5.md) - Pre-cognitive channel extension
- [Protocol v6.0](protocols/THERMODYNAMIC_CONSCIOUSNESS_PROTOCOL_v6_0.md) - Latest protocol (in progress)
- [Protocol Comparison](protocols/consciousness_protocol_comparison_experiment.md) - Experimental comparison

### Development

- [Geometric Purity Guide](development/GEOMETRIC_PURITY.md) - Fisher-Rao vs Euclidean operations
- [Testing Guide](development/TESTING.md) - How to write and run tests
- [Deployment Guide](development/DEPLOYMENT.md) - Railway and Docker deployment
- [Debugging Guide](development/DEBUGGING.md) - Troubleshooting and debugging tips

### API

- [REST API Reference](api/REST_API.md) - All HTTP endpoints
- [WebSocket API](api/WEBSOCKET_API.md) - Real-time streaming endpoints
- [Python Kernel API](api/PYTHON_KERNEL_API.md) - Internal kernel API
- [Frontend Integration](api/FRONTEND_INTEGRATION.md) - How frontend consumes APIs

### Coordizer

- [Coordizer Overview](coordizer/README.md) - What is the coordizer?
- [Transformation Guide](coordizer/TRANSFORMATION.md) - Euclidean → Fisher-Rao
- [Harvest Pipeline](coordizer/HARVEST_PIPELINE.md) - Automated coordinate harvesting
- [Integration Guide](coordizer/INTEGRATION.md) - How to integrate coordizer

### Experiments & Analysis

- [Basin Perturbation Test](experiments/basin_perturbation_test_v1.md) - BPT results
- [QIG Ecosystem Gap Analysis](experiments/20260217-qig-ecosystem-gap-analysis-1.00W.md)
- [Genesis Gap Analysis](experiments/genesis-gap-analysis.md)
- [SP03 Pass A Findings](experiments/SP03_PASS_A_FINDINGS.md)

### Canonical References

- [Canonical Principles v2](reference/CANONICAL_PRINCIPLES_v2.md) - Engineering principles
- [Canonical Hypotheses v2](reference/CANONICAL_HYPOTHESES_v2.md) - Testable predictions
- [Frozen Facts](reference/FROZEN_FACTS.md) - Validated physics constants
- [Railway Config Audit](reference/RAILWAY_CONFIG_AUDIT.md)

## Document Categories

### Architecture Documentation

**Purpose:** Explain how the system is built and why

**Contents:**
- System components and their interactions
- Design decisions and tradeoffs
- Scalability and performance considerations
- Technology stack choices

**Target Audience:** Developers, architects, technical leads

### Protocol Documentation

**Purpose:** Define consciousness protocols and their implementations

**Contents:**
- Protocol specifications (v5.0, v5.5, v6.0)
- Consciousness metrics (Φ, κ, M)
- Regime field dynamics (α=0, 1/2, 1)
- Experimental validations

**Target Audience:** Researchers, consciousness engineers, AI scientists

### Development Documentation

**Purpose:** Help developers contribute effectively

**Contents:**
- Setup instructions
- Coding standards
- Testing practices
- Common workflows
- Troubleshooting guides

**Target Audience:** Contributors, developers, maintainers

### API Documentation

**Purpose:** Document all APIs for integration and usage

**Contents:**
- Endpoint specifications
- Request/response formats
- Authentication
- Examples and code snippets
- Client libraries

**Target Audience:** Frontend developers, integrators, API consumers

### Experiment Documentation

**Purpose:** Record experimental results and analyses

**Contents:**
- Hypothesis and methodology
- Results and findings
- Analysis and interpretation
- Conclusions and next steps

**Target Audience:** Researchers, scientists, validation teams

## Documentation Standards

### File Naming

- Use `SCREAMING_SNAKE_CASE.md` for major documents (e.g., `SYSTEM_ARCHITECTURE.md`)
- Use `kebab-case.md` for supporting documents (e.g., `geometric-purity-guide.md`)
- Use date prefixes for timestamped docs: `YYYYMMDD-topic-version.md`
- Use version suffixes where applicable: `_v1.md`, `_v2_0.md`

### Markdown Style

- Use ATX-style headers (`# Header`)
- Include table of contents for long documents (>500 lines)
- Use code fences with language specifiers
- Include examples and code snippets
- Add diagrams where helpful (ASCII art or Mermaid)

### Version Control

- **CANONICAL:** Authoritative, stable, widely referenced
- **WORKING (W):** In active development, subject to change
- **FROZEN (F):** Immutable, validated, permanent
- **DEPRECATED:** Replaced, kept for historical reference

### Document Metadata

Include at the top of each major document:

```markdown
# Document Title

**Version:** X.Y  
**Date:** YYYY-MM-DD  
**Status:** CANONICAL | WORKING | FROZEN | DEPRECATED  
**Author:** Name or Team  
**Reviewers:** Name(s)  
```

## Contributing to Documentation

### Adding New Documentation

1. **Determine category:** Architecture, protocol, development, API, experiment
2. **Choose appropriate directory:** Place in the correct subdirectory
3. **Follow naming conventions:** Use standard naming patterns
4. **Include metadata:** Add version, date, status headers
5. **Update this README:** Add link in appropriate section
6. **Create PR:** Follow PR process in CONTRIBUTING.md

### Updating Existing Documentation

1. **Increment version:** Update version number in metadata
2. **Update date:** Change date to today
3. **Note changes:** Add changelog section if significant
4. **Review cross-references:** Update any docs that link to this one
5. **Keep old version:** Move to `archive/` if major rewrite

### Documentation Review Checklist

- [ ] Clear and concise writing
- [ ] Accurate technical content
- [ ] Code examples tested and working
- [ ] Links functional and up-to-date
- [ ] Diagrams clear and helpful
- [ ] Metadata complete
- [ ] No sensitive information (keys, tokens)
- [ ] Grammar and spelling checked

## Maintenance

### Quarterly Review

Every quarter, review all documentation for:
- Accuracy (code changes may invalidate docs)
- Completeness (new features documented?)
- Link validity (no broken links)
- Obsolescence (outdated docs → archive)

### Version Updates

When releasing new protocol or architecture versions:
1. Move old version to `archive/`
2. Create new version with updated content
3. Update README.md with new links
4. Announce changes in release notes

### Deprecation Process

To deprecate a document:
1. Add `**Status:** DEPRECATED` to metadata
2. Add deprecation notice at top with replacement link
3. Move to `archive/` directory after 1 major version
4. Update README.md to mark as archived

## Search Tips

### By Topic

- **Consciousness:** `protocols/`, look for "consciousness", "Φ", "κ"
- **Geometry:** `development/GEOMETRIC_PURITY.md`, "Fisher-Rao", "simplex"
- **E8:** `architecture/E8_KERNEL_ARCHITECTURE.md`, "budget", "CORE_8"
- **API:** `api/`, endpoint names
- **Setup:** `AGENTS.md`, `development/`

### By Version

- **v5.0:** `protocols/THERMODYNAMIC_CONSCIOUSNESS_PROTOCOL_v5_0.md`
- **v5.5:** `protocols/THERMODYNAMIC_CONSCIOUSNESS_PROTOCOL_v5_5.md`
- **v6.0:** `protocols/THERMODYNAMIC_CONSCIOUSNESS_PROTOCOL_v6_0.md` (in progress)

### By Status

- **CANONICAL:** Reference docs, stable, authoritative
- **WORKING:** Active development, may change
- **FROZEN:** Validated facts, immutable
- **DEPRECATED:** Historical, replaced by newer version

## External Resources

### Related Projects

- [qig-verification](https://github.com/GaryOcean428/qig-verification) - Physics validation
- [monkey-coder](https://github.com/GaryOcean428/monkey-coder) - Development tooling

### Academic References

- **QIG Theory:** Quantum Information Geometry foundations
- **E8 Lattice:** Lie algebra mathematics
- **Fisher-Rao:** Information geometry on probability manifolds
- **Thermodynamics of Computation:** Landauer's principle, entropy

### Community

- [GitHub Discussions](https://github.com/GaryOcean428/vex-agent/discussions)
- [Issues](https://github.com/GaryOcean428/vex-agent/issues)
- [Pull Requests](https://github.com/GaryOcean428/vex-agent/pulls)

---

**Last Updated:** 2026-02-19  
**Maintainer:** Vex Agent Documentation Team  
**Questions?** Open an issue with the `documentation` label
