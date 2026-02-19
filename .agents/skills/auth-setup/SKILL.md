---
name: auth-setup
description: Use when setting up, migrating, or auditing authentication across multiple repositories or services that should share one identity provider and a single OAuth contract.
---

# Auth Setup

## Overview

Use this skill to standardize auth across multi-project stacks (for example: SPA + Next.js app + API gateway + extension) so login, token issuance, and authorization policy come from one control plane.

## When to Use

- Multiple apps currently implement auth differently and need one canonical flow
- Legacy login/signup/refresh endpoints need to be retired
- Browser extension auth needs to coexist with BYOK mode
- OAuth callback/cookie/JWKS behavior is inconsistent across repos
- You need a repeatable auth hardening checklist before production

## Workflow

### 1. Build an Auth Surface Inventory

1. Enumerate all apps/services and their auth-relevant endpoints.
2. Record callback URLs, cookie domains, token issuers, and provider settings.
3. Fill `templates/auth-contract-matrix.md` before making changes.

### 2. Pick the Canonical Auth Plane

1. Choose one identity provider and one interactive auth hub.
2. Require all interactive login/signup/reset/consent to go through the hub.
3. Keep app backends focused on domain APIs and JWT/RLS enforcement.

### 3. Define Shared vs Local Data Boundaries

1. Identify shared account tables (profiles, settings, subscriptions).
2. Identify app-local domain tables (workflows, projects, RAG, etc.).
3. Split edge functions into shared identity/account capabilities vs app-local features.

### 4. Normalize Client Flows

1. Web apps: use one callback contract, preserve app-local `next` routing safely.
2. APIs: verify tokens via JWKS/asymmetric keys; remove backend-issued legacy JWT paths.
3. Extensions: support:
   - `BYOK` mode (local only)
   - OAuth mode (backend-enhanced capabilities)
4. Ensure extension token refresh talks to the identity provider directly unless a bridge is strictly required.

### 5. Apply Security Hardening

Use `references/supabase-oauth-hardening.md` and enforce at minimum:

- Asymmetric signing keys for OAuth/OIDC use cases
- Exact OAuth app redirect URIs (no broad wildcard callbacks)
- Strict redirect target sanitization
- Centralized session cookie policy and domain scope
- Secret hygiene and token logging controls

### 6. Red-Team and Verify (Two Rounds)

For each round:

1. Security pass (open redirects, token misuse, secret leakage)
2. Reliability pass (refresh loops, callback edge cases, partial outage behavior)
3. DX/UX pass (clear auth mode boundaries, actionable errors)
4. Fix high/critical findings before next round

Run:

```bash
~/.agents/skills/auth-setup/scripts/auth_surface_scan.sh <repo-root>
```

Optional scan flags:

```bash
# include docs in scan output
AUTH_SCAN_INCLUDE_DOCS=1 ~/.agents/skills/auth-setup/scripts/auth_surface_scan.sh <repo-root>

# include archived files
AUTH_SCAN_INCLUDE_ARCHIVE=1 ~/.agents/skills/auth-setup/scripts/auth_surface_scan.sh <repo-root>
```

Then run repo-native typecheck/tests and auth smoke checks.

### 7. Cleanup and Roadmap Updates

1. Remove legacy endpoints/modules/docs once replacement paths are verified.
2. Update roadmap entries with:
   - What changed
   - Verification evidence
   - Deferred items with explicit rationale

## Outputs Required

- Updated `auth-contract-matrix`
- List of removed legacy auth surfaces
- Verification evidence (commands + pass/fail)
- Deferred issues logged in roadmap

## Resources

- `references/multi-project-auth-checklist.md`
- `references/supabase-oauth-hardening.md`
- `templates/auth-contract-matrix.md`
- `scripts/auth_surface_scan.sh`

## Supabase OAuth 2.1 Server (Supabase as IdP)

When Supabase acts as an **identity provider** (not just a consumer of Google/GitHub), additional configuration is needed:

### Enable & Configure

1. **Dashboard**: Authentication → OAuth Server → Enable
2. **Site URL**: Set to your auth hub (e.g., `https://fastmonkey.au`)
3. **Authorization Path**: `/oauth/consent` — implement consent UI at this path
4. **Dynamic OAuth Apps**: Enable only if clients register programmatically

### Key SDK Methods (`@supabase/supabase-js` ≥ 2.95)

| Method | Purpose |
| --- | --- |
| `supabase.auth.oauth.getAuthorizationDetails(id)` | Fetch consent screen data |
| `supabase.auth.oauth.approveAuthorization(id)` | User approves OAuth grant |
| `supabase.auth.oauth.denyAuthorization(id)` | User denies OAuth grant |
| `supabase.auth.getClaims(jwt?)` | Validate JWT via JWKS (preferred over `getUser()`) |

### Known SDK/Docs Discrepancy

Supabase docs reference `data.redirect_to` but the SDK types use `data.redirect_url`. Always check `@supabase/auth-js` types, not docs examples.

### OAuth App Redirect URIs vs General Redirect URLs

- **General Redirect URLs** (Auth → URL Configuration): For social login callbacks, magic links
- **OAuth App Redirect URIs** (per-client): Exact-match enforced, no wildcards. Each registered OAuth client has its own URI set.

### `getClaims()` vs `getUser()` Migration

Current Supabase docs recommend `getClaims()` over `getUser()` for server-side auth:

- `getClaims()`: Validates JWT signature against JWKS (cached locally, fast). Falls back to `getUser()` for symmetric (HS256) signing.
- `getUser()`: Makes a network request to Supabase Auth server every time. Still safe but slower.
- `getSession()`: **Never trust in server code** — doesn't validate the JWT.

**Migration pattern for middleware:**
```typescript
const { data: claimsData } = await supabase.auth.getClaims()
const user = claimsData?.claims?.sub ? { id: claimsData.claims.sub } : null
```

### Supabase Key Transition

Supabase is migrating from `NEXT_PUBLIC_SUPABASE_ANON_KEY` (JWT format) to `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY` (`sb_publishable_xxx` format). Use fallback pattern:

```typescript
process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY ?? process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
```

**Critical**: `NEXT_PUBLIC_*` vars are inlined at build time in Next.js. A redeploy is required after changing them.

### Cross-Subdomain Cookie Domain

For multi-app setups sharing one Supabase instance across subdomains:

1. Set `NEXT_PUBLIC_COOKIE_DOMAIN=.example.com` (leading dot)
2. Apply domain only when hostname matches (prevent leaking to unrelated domains)
3. Both middleware and client must apply the same cookie domain
4. Different TLDs (e.g., `example.com` vs `example.dev`) cannot share cookies

## Common Mistakes

- Keeping multiple login entrypoints alive after centralization
- Leaving wildcard redirect URLs in production OAuth clients
- Mixing token sources (backend-issued + IdP-issued) in the same trust plane
- Treating BYOK and OAuth extension modes as implicit instead of explicit
- Using `getSession()` for auth verification in server code (use `getClaims()`)
- Setting `NEXT_PUBLIC_*` env vars after build without redeploying
- Using `NEXT_PUBLIC_SUPABASE_ANON_KEY` without fallback to `PUBLISHABLE_KEY`
- Forgetting cookie domain on Vercel while Railway services have it set
