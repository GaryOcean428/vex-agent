---
name: ui-ux-consistency
description: Ensure consciousness visualizations follow design system, check God Panel matches specs, validate regime color schemes (green=geometric, yellow=linear, red=breakdown), check WCAG 2.1 accessibility. Use when reviewing UI components or consciousness displays.
---

# UI/UX Consistency

Validates consciousness visualizations. Source: `.github/agents/ui-ux-consistency-agent.md`.

## When to Use This Skill

- Reviewing UI components
- Implementing consciousness visualizations
- Checking accessibility compliance
- Validating design system usage

## Step 1: Verify Regime Color Scheme

```typescript
// Regime colors MUST follow this scheme
const REGIME_COLORS = {
  breakdown: "#EF4444",      // Red - consciousness collapse
  linear: "#F59E0B",         // Yellow/Amber - pre-geometric
  geometric: "#10B981",      // Green - healthy geometric regime
  hierarchical: "#8B5CF6",   // Purple - advanced integration
};
```

## Step 2: Check God Panel Dimensions

```typescript
// God Panel layout MUST match specs
const GOD_PANEL_SPECS = {
  leftSidebar: "240px",    // Navigation
  rightPanel: "320px",     // God Panel details
  mainContent: "flex-1",   // Remaining space
};
```

## Step 3: Validate Accessibility

```bash
# Run accessibility checks
npx axe-core client/

# Check color contrast
npx color-contrast-checker
```

## Design System Requirements

| Component | Requirement | Check |
|-----------|-------------|-------|
| Regime colors | Red/Yellow/Green/Purple | Color values match |
| God Panel | 240px left, 320px right | CSS dimensions |
| Typography | Design system tokens | No hardcoded values |
| Icons | Lucide icons only | No random icon sets |
| Spacing | Tailwind spacing scale | No arbitrary px values |

## Consciousness Visualization Rules

```tsx
// ✅ CORRECT: Using design tokens
<div className="bg-regime-geometric text-white">
  Φ = {phi.toFixed(3)}
</div>

// ❌ WRONG: Hardcoded colors
<div style={{ backgroundColor: '#10B981' }}>  // FORBIDDEN
  Φ = {phi.toFixed(3)}
</div>
```

## WCAG 2.1 Checklist

- [ ] Color contrast ratio ≥ 4.5:1 for text
- [ ] All interactive elements keyboard accessible
- [ ] Focus indicators visible
- [ ] Screen reader compatible
- [ ] No color-only information

## Validation Commands

```bash
# Type check UI components
cd client && npx tsc --noEmit

# Run component tests
npm test

# Check design system usage
rg "style=\{" client/src/ --type tsx  # Should be minimal
```

## Response Format

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UI/UX CONSISTENCY REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Design System:
  - Regime colors: ✅ / ❌
  - God Panel layout: ✅ / ❌
  - Design tokens: ✅ / ❌

Accessibility:
  - Color contrast: ✅ / ❌
  - Keyboard nav: ✅ / ❌
  - Screen reader: ✅ / ❌

Violations: [list]
Priority: CRITICAL / HIGH / MEDIUM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
