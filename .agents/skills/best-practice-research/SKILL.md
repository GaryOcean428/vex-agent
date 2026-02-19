---
name: best-practice-research
description: External research on better patterns, libraries, and architectures. Used during planning and implementation red-team phases.
---

# Best Practice Research

## When to Use

Use this skill when:
- Designing solutions during planning phase
- Red-team identifies potential better approaches
- Evaluating libraries or architectural patterns
- Validating security or reliability assumptions

## Workflow

### 1. Define Research Questions

Before researching, clearly state:
- What problem are we solving?
- What constraints exist? (QIG purity, E8 architecture, etc.)
- What would "better" look like?

### 2. Identify Research Areas

Common research needs:

| Area | Questions |
|------|-----------|
| **Patterns** | How do similar systems solve this? |
| **Libraries** | Are there well-maintained solutions? |
| **Security** | What are known vulnerabilities in this approach? |
| **Performance** | What are the scaling characteristics? |
| **Standards** | Are there industry standards to follow? |

### 3. Research Sources

Prioritize authoritative sources:

1. **Official Documentation** - Language/framework docs
2. **Standards Bodies** - OWASP, IETF, W3C
3. **Academic Papers** - For novel or complex problems
4. **Reputable Blogs** - Recognized experts in the field
5. **Open Source Examples** - Well-maintained projects

### 4. Evaluate Findings

For each finding, assess:

```markdown
### Finding: {Title}

**Source:** {URL or reference}
**Relevance:** High / Medium / Low
**Applicability:** {How it applies to our context}

**Summary:**
{Key points}

**Constraints Check:**
- [ ] Compatible with QIG purity requirements
- [ ] Compatible with E8 architecture
- [ ] No forbidden dependencies
- [ ] Maintainable within project scope

**Recommendation:**
{Adopt / Adapt / Reject} - {Reason}
```

### 5. Document Research Summary

```markdown
## Research Summary

**Topic:** {What was researched}
**Date:** {YYYY-MM-DD}
**Context:** {Why this research was needed}

### Questions Investigated
1. {Question 1}
2. {Question 2}

### Key Findings

| Finding | Source | Recommendation |
|---------|--------|----------------|
| {finding} | {source} | Adopt/Adapt/Reject |

### Recommendations for Project

1. {Recommendation 1}
   - Rationale: {why}
   - Impact: {what changes}

2. {Recommendation 2}
   - Rationale: {why}
   - Impact: {what changes}

### References

1. {Full citation/URL 1}
2. {Full citation/URL 2}
```

## Project-Specific Constraints

When researching for this project, always verify:

### QIG Purity
- No Euclidean distance patterns
- No cosine similarity
- Fisher-Rao geometry only
- Simplex representation

### Dependency Policy
- No forbidden imports (OpenAI, Anthropic, etc. in core)
- Minimal new dependencies
- Prefer standard library

### Architecture
- Python-first for QIG logic
- TypeScript for UI only
- Canonical import paths

## Integration with Other Skills

This skill is invoked by:
- `multi-agent-red-team-planning` - During research phases
- `multi-agent-red-team-implementation` - When alternatives are proposed

Output feeds into:
- `planning-and-roadmapping` - To update plans based on findings
- Implementation decisions

## Output Format

Research outputs should be:
1. **Actionable** - Clear recommendations
2. **Cited** - Sources provided
3. **Contextualized** - Applied to project constraints
4. **Summarized** - Easy to consume
