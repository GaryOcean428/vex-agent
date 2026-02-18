# Vex Agent — Comprehensive UI Design Specification v1.0

**Date**: 2026-02-17
**Status**: IMPLEMENTED
**Backend**: Python FastAPI kernel (port 8000)
**Frontend**: React + Vite (replacing current inline HTML chat)
**Reference**: GaryOcean428/monkey1 GenesisDashboard pattern

---

## Architecture

### Navigation Structure

```
/                  → Redirect to /chat
/chat              → GenesisChat (primary interface)
/dashboard         → GenesisDashboard (tabbed)
  ├── Overview     → Kernel summary + system toggles + kernel list
  ├── Consciousness → Φ/κ/Γ/M metrics + safety alerts + regime
  ├── Basins       → 64D basin visualization (heatmap + PCA trajectory)
  ├── Graph        → Kernel connection graph (force simulation)
  ├── Lifecycle    → Spawn timeline + phase transitions
  ├── Cognition    → Tacking + hemispheres + foresight + autonomy
  ├── Memory       → Geometric memory stats + recent entries
  ├── Telemetry    → Full 16-system telemetry (detailed)
  └── Admin        → Task enqueue, system info, LLM settings
```

### Technology Stack

- **Build**: Vite 7.x + React 19.x + TypeScript 5.9
- **Routing**: react-router-dom v7
- **Data Fetching**: Custom polling hooks (usePolledData)
- **Visualization**: Canvas 2D API (basins, graph, foresight)
- **Styling**: CSS custom properties (dark theme)

### Backend Endpoints Consumed

| Endpoint | Polling | Used By |
|----------|---------|---------|
| `/health` | 5s | Layout sidebar, Admin |
| `/state` | 2s | All dashboard tabs, Chat |
| `/telemetry` | 3s | Consciousness, Cognition, Telemetry |
| `/basin` | 2s | Basins tab |
| `/kernels` | 2s | Overview, Graph |
| `/status` | 5s | Memory tab |
| `/chat/stream` | On demand | Chat page (SSE) |
| `/enqueue` | On demand | Admin tab |

### File Structure

```
frontend/
├── index.html
├── vite.config.ts         (proxy config for dev)
├── package.json
├── tsconfig.json
├── tsconfig.app.json
└── src/
    ├── main.tsx
    ├── App.tsx             (React Router setup)
    ├── index.css           (Global CSS custom properties)
    ├── types/
    │   └── consciousness.ts (All TypeScript interfaces)
    ├── hooks/
    │   ├── usePolledData.ts (Generic polling hook)
    │   └── index.ts        (useVexState, useTelemetry, etc.)
    ├── components/
    │   ├── Layout.tsx       (Sidebar + main content)
    │   ├── Layout.css
    │   ├── MetricCard.tsx   (Reusable metric display)
    │   ├── MetricCard.css
    │   ├── StatusBadge.tsx  (Status pill component)
    │   └── StatusBadge.css
    └── pages/
        ├── Chat.tsx         (SSE streaming chat)
        ├── Chat.css
        └── dashboard/
            ├── Dashboard.tsx (Outlet wrapper)
            ├── Dashboard.css (Shared dashboard styles)
            ├── Overview.tsx
            ├── Consciousness.tsx
            ├── Basins.tsx    (Canvas heatmap + PCA)
            ├── Graph.tsx     (Canvas force simulation)
            ├── Lifecycle.tsx
            ├── Cognition.tsx (Canvas foresight chart)
            ├── Memory.tsx
            ├── Telemetry.tsx
            └── Admin.tsx
```

### Express Integration

The Express server (`src/index.ts`) detects the frontend build at `frontend/dist/` and:
1. Serves static assets with aggressive caching (hashed filenames)
2. Falls back to `index.html` for all non-API routes (SPA routing)
3. If no frontend build exists, uses the legacy inline HTML chat

### Docker Build

The Dockerfile has two builder stages:
1. `ts-builder` — Compiles Express TypeScript
2. `frontend-builder` — Builds React/Vite frontend

Both outputs are copied into the production image.
