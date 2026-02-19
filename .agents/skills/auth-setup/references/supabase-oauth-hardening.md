# Supabase OAuth Hardening Notes

Practical guardrails for Supabase as OAuth 2.1 / OIDC provider in multi-app environments.

## 1. Identity Provider Baseline

- Use one canonical Supabase project for shared identity if tenants are shared.
- Keep interactive auth UX centralized (auth hub) and let product apps consume tokens.
- For OAuth/OIDC provider scenarios, prefer asymmetric signing keys for token verification via JWKS.

## 2. Redirect Controls

- OAuth app redirect URIs must be exact callback URLs, per environment/client.
- Avoid broad wildcard callback domains in production.
- Treat extension redirect URLs as untrusted input and validate host/protocol strictly.
- Sanitize all `next`/`redirect` params to known targets.

## 3. Token and Session Rules

- Keep one source of truth for access/refresh tokens.
- Avoid dual token systems (legacy backend JWT + Supabase JWT) unless temporary migration bridge is documented and time-boxed.
- Ensure refresh behavior is deterministic:
  - preferred: direct Supabase token endpoint
  - bridge fallback only when required by platform constraints

## 4. RLS and Claims

- RLS should enforce user-level access and optional client-level scoping.
- If custom claims are used, define and document schema and validation expectations.
- Audit all shared tables for consistent `auth.uid()` usage patterns.

## 5. Operational Controls

- Add endpoint sweeps in CI for banned legacy auth routes.
- Add secret hygiene checks for auth tokens/keys in tracked files.
- Keep auth smoke scripts for:
  - login/callback
  - refresh
  - guarded API access
  - logout

## 6. OAuth 2.1 Server Endpoints (Supabase as IdP)

When Supabase is configured as an OAuth 2.1 identity provider (Authentication → OAuth Server → Enable):

| Endpoint | URL |
|---|---|
| **OAuth Discovery** | `https://<ref>.supabase.co/.well-known/oauth-authorization-server/auth/v1` |
| **OIDC Discovery** | `https://<ref>.supabase.co/auth/v1/.well-known/openid-configuration` |
| **JWKS** | `https://<ref>.supabase.co/auth/v1/.well-known/jwks.json` |
| **Authorization** | `https://<ref>.supabase.co/auth/v1/oauth/authorize` |
| **Token** | `https://<ref>.supabase.co/auth/v1/oauth/token` |
| **UserInfo** | `https://<ref>.supabase.co/auth/v1/oauth/userinfo` |

OAuth app redirect URIs are **exact-match** enforced (separate from general redirect URL allow-list).

> ⚠️ **Common mistake**: The authorization endpoint is `/auth/v1/oauth/authorize` (with `/oauth/`), NOT `/auth/v1/authorize`. The JWKS endpoint is `/auth/v1/.well-known/jwks.json` (under `/auth/v1/`), NOT at the root.

## 6a. MCP Resource Server Pattern

When building an MCP server (resource server) that uses Supabase as the identity provider, the pattern is:

### Token Validation (Resource Server — e.g., mcp-browser-http-server)

Use `jose` with `createRemoteJWKSet` — **never use the JWT secret** (that's symmetric/HS256 only):

```ts
import { createRemoteJWKSet, jwtVerify } from 'jose';

const SUPABASE_URL = process.env.SUPABASE_URL!; // e.g. https://kxdaxwvxaonnvjmqfvtj.supabase.co
const JWKS = createRemoteJWKSet(
  new URL(`${SUPABASE_URL}/auth/v1/.well-known/jwks.json`)
);

async function validateToken(token: string) {
  const { payload } = await jwtVerify(token, JWKS, {
    issuer: `${SUPABASE_URL}/auth/v1`,
    audience: 'authenticated',
  });
  return {
    userId: String(payload.sub),
    email: payload.email as string | undefined,
    clientId: payload.client_id as string | undefined, // Present for OAuth 2.1 client tokens
  };
}
```

### OAuth Protected Resource Metadata (RFC 9728 — required for MCP 2025-03-26)

The MCP server MUST serve this endpoint so MCP clients (Claude Code, Gemini CLI) can discover the auth server:

```ts
// GET /.well-known/oauth-protected-resource
app.get('/.well-known/oauth-protected-resource', (_req, res) => {
  res.json({
    resource: process.env.MCP_SERVER_PUBLIC_URL, // e.g. https://one.fastmonkey.au
    authorization_servers: [process.env.SUPABASE_URL], // e.g. https://kxdaxwvxaonnvjmqfvtj.supabase.co
    bearer_methods_supported: ['header'],
    scopes_supported: ['openid', 'email', 'profile'],
  });
});
```

### 401 Response (MCP spec requirement)

All 401 responses must include the WWW-Authenticate header pointing to the resource metadata:

```ts
res.setHeader(
  'WWW-Authenticate',
  `Bearer realm="monkey-mcp", resource_metadata="${process.env.MCP_SERVER_PUBLIC_URL}/.well-known/oauth-protected-resource"`
);
res.status(401).json({ error: 'Authentication required' });
```

### Two Distinct Auth Patterns (Never Conflate)

| Client | Pattern | Result |
|---|---|---|
| **Browser Extension** | Standard Supabase social auth (Google/GitHub) via `chrome.identity.launchWebAuthFlow` | Supabase JWT (standard session, no `client_id` claim) |
| **Claude Code / Gemini CLI / MCP clients** | Supabase OAuth 2.1 Server PKCE + consent screen at `/oauth/consent` | Supabase JWT with `client_id` claim |

Both tokens are RS256 JWTs validated identically via JWKS. The `client_id` claim presence differentiates them.

### MCP Client Discovery Flow (Claude Code auto-auth)

```
1. Claude Code → POST /mcp → 401 + WWW-Authenticate header
2. Claude Code → GET /.well-known/oauth-protected-resource
   → { authorization_servers: ["https://kxdaxwvxaonnvjmqfvtj.supabase.co"] }
3. Claude Code → GET https://kxdaxwvxaonnvjmqfvtj.supabase.co/.well-known/oauth-authorization-server/auth/v1
   → Supabase's full OAuth discovery document
4. Claude Code → registers dynamically (if enabled) or uses pre-registered client_id
5. Claude Code → opens browser → Supabase redirects to https://fastmonkey.au/oauth/consent
6. User approves → Supabase issues JWT with client_id claim
7. Claude Code → POST /mcp with Bearer JWT → validated via JWKS → success
```

### No JWT Secret Needed on Resource Server

The resource server (MCP server) only needs:
- `SUPABASE_URL` — to build the JWKS endpoint URL
- `MCP_SERVER_PUBLIC_URL` — to populate the resource metadata

It does NOT need `SUPABASE_JWT_SECRET` or `SUPABASE_SERVICE_ROLE_KEY` for token validation.

## 7. Server-Side JWT Verification

- **Prefer `getClaims()`** over `getUser()` — validates JWT via cached JWKS (fast, no network request per call)
- `getClaims()` **requires asymmetric signing (RS256/ES256)** — will throw `AuthApiError` on every request with HS256
- `getUser()` still works (makes network call to Supabase) but is slower; acceptable fallback if HS256 is still active
- **Never use `getSession()` for auth verification** in server code — it doesn't validate the JWT
- For middleware (runs every request), `getClaims()` is significantly faster and avoids token-refresh race conditions
- **Confirmed working** (Feb 2026): RS256 active on `kxdaxwvxaonnvjmqfvtj`, `getClaims()` validates locally via JWKS

## 8. Supabase Key Transition

- Old: `NEXT_PUBLIC_SUPABASE_ANON_KEY` (JWT format: `eyJ...`)
- New: `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY` (`sb_publishable_xxx`)
- Both work during transition; code should prefer new name with old as fallback
- `NEXT_PUBLIC_*` vars are build-time in Next.js — **redeploy required** after changes

## 9. Cross-Subdomain Session Sharing

- Set cookie domain to `.example.com` (leading dot) for `*.example.com` subdomains
- Apply domain conditionally based on request hostname (avoid leaking to unrelated domains)
- Both server middleware and browser client must apply the same domain
- Different TLDs cannot share cookies (`.example.com` ≠ `.example.dev`)
- `SameSite=Lax` is correct for top-level navigation redirects between subdomains

## 10. Migration Discipline

- Sequence migrations so dependent `ALTER` statements cannot fail on clean environments.
- Validate schema dependencies before deployment.
- Update docs and runbooks in the same change-set as auth surface changes.

## 11. Edge Function "Verify JWT" After RS256 Migration

- Supabase Edge Functions have a built-in "Verify JWT" toggle (enabled by default).
- This built-in verification uses the **old HS256 JWT secret** — it does NOT auto-update after RS256 migration.
- After migrating to RS256, **all edge functions with "Verify JWT" enabled will return 401** for every request.
- **Fix**: Disable "Verify JWT" in Supabase Dashboard → Edge Functions → each function.
- This is safe if functions implement their own auth via `authenticateRequest()` / `supabase.auth.getUser()`.
- The Supabase docs explicitly state: _"Edge Functions with 'Verify JWT' setting must be turned OFF after migration."_

## 12. Middleware Callback Skip Pattern

- SSR middleware that refreshes tokens (via `getClaims()` or `getUser()`) must **skip the `/auth/callback` route**.
- `exchangeCodeForSession()` in the callback sets its own cookies; middleware cookie writes can overwrite them.
- This causes a race condition where the session cookie from the callback is lost, resulting in redirect loops.
- Pattern:

  ```typescript
  if (request.nextUrl.pathname === '/auth/callback') {
    return NextResponse.next({ request })
  }
  ```

- Apply this early in `updateSession()` before creating the Supabase client.
