# Auth Contract Matrix — Monkey Ecosystem

Filled with live-verified values (Feb 2026). Supabase project: `kxdaxwvxaonnvjmqfvtj`.

| App/Client | Framework/Runtime | Deploy Target | Login Entry URL | Callback URL(s) | Session Storage | Token Issuer | Refresh Path | Auth Mode(s) | Notes |
|---|---|---|---|---|---|---|---|---|---|
| **monkey-oauth** (auth hub) | Next.js 15.2.6, Node 24.x | Vercel (`prj_6Ws3...8cTF`) | `fastmonkey.au/auth/login` | `fastmonkey.au/auth/callback` | Supabase cookie (domain `.fastmonkey.au`) | Supabase | Supabase SSR auto-refresh | OAuth (Google, GitHub), email/password | Auth hub — all interactive login goes here |
| **monkey1** (Monkey One) | React Router SPA + Express API GW + FastAPI backend | Railway (12 services) | Redirects to auth hub: `fastmonkey.au/auth/login?next=one` | `one.fastmonkey.au/auth/callback` | Supabase cookie (domain `.fastmonkey.au`) | Supabase | Supabase client auto-refresh | SSO via auth hub redirect | `VITE_COOKIE_DOMAIN=.fastmonkey.au` |
| **monkey-coder** | Next.js frontend + FastAPI backend | Railway (9 services) | Redirects to auth hub: `fastmonkey.au/auth/login?next=coder` | `coder.fastmonkey.au/auth/callback` | Supabase cookie (domain `.fastmonkey.au`) + backend JWT bridge | Supabase (primary) + backend JWT (bridge) | Supabase + `exchangeSupabaseToken()` | SSO via auth hub redirect | Legacy JWT bridge still active (phase 2 removal) |
| **monkey1 API gateway** | Express | Railway | n/a | n/a | n/a | Supabase (JWKS verify) | n/a | JWT verification | `extensionAuth` middleware: BYOK key or Supabase JWT |
| **monkey-coder backend** | FastAPI | Railway | n/a | n/a | n/a | Supabase (JWKS verify) + backend JWT | n/a | JWT verification + token exchange | JWKS endpoint for asymmetric verification |
| **browser extension** | Chrome MV3 | Chrome Web Store | Extension bridge: `one.fastmonkey.au/auth/extension-bridge` | `https://<ext-id>.chromiumapp.org/` | `chrome.storage.local` | Supabase (OAuth mode), none (BYOK mode) | Extension-managed | byok / oauth | Dual mode: BYOK (local API keys) or OAuth (backend-enhanced) |

## Shared Tables

- `auth.users`: Canonical identity, managed by Supabase Auth
- `public.profiles`: Display name, avatar, bio, username, role — shared across all apps
- `public.user_settings`: App-agnostic preferences
- `billing.*`: Subscription tier, token usage, billing cycles — shared via edge functions

## App-Local Tables

- **monkey1**: Workspaces, projects, workflows, RAG indices, runtime configs
- **monkey-coder**: Code sessions, file trees, terminal history, deployment configs
- **Extension**: Local BYOK keys in `chrome.storage.local` (not in Supabase)

## Shared Edge Functions

- `account-profile`: Profile CRUD
- `account-security`: Audit events, MFA factors, session management
- `billing-summary`: Subscription and usage data
- `billing-usage-rollup`: Per-product usage aggregation
- `manage-api-keys`: API key CRUD

## App-Local Edge Functions

- monkey1: Workspace-specific functions (if any)
- monkey-coder: Code execution, deployment triggers

## Security Decisions

- **JWT signing algorithm**: RS256 (active, migrated from HS256 Feb 2026); target ES256 (per AGENTS.md roadmap)
- **JWKS verification strategy**: `getClaims()` in middleware (cached JWKS, validated working with RS256)
- **Edge Function "Verify JWT"**: Must be **disabled** after RS256 migration (built-in verification uses old HS256 secret)
- **Redirect allow-list policy**: Exact URLs in Supabase dashboard; `next` param sanitized to known targets in callback route
- **Cookie domain policy**: `.fastmonkey.au` on all services; hostname-conditional application (doesn't apply on `monkey-one.dev`)
- **Secret management policy**: Vercel env vars (sensitive flag), Railway env vars, no `.env.local` in repos

## Verification Evidence

- **Endpoint sweep**: `auth_surface_scan.sh` — pending (update scan patterns first)
- **Typecheck/build**: monkey-oauth `next build` passes (exit 0, only img lint warnings)
- **Vercel deployment**: `dpl_7xg286qiHcRCwwph4Bf3mzUU3zfD` READY (commit `81e87a9`)
- **Railway env vars**: Verified via `mcp4_list-variables` — all cookie domains and Supabase keys set
- **Red-team round 1**: PUBLISHABLE_KEY migration + getClaims() applied; error handling added to middleware
