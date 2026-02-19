# Multi-Project Auth Checklist

Use this checklist when converging multiple repos/services onto one auth system.

## A. Discovery

- [ ] Enumerate all auth entrypoints across repos (`login`, `signup`, `refresh`, `oauth/callback`, `extension bridge`)
- [ ] Enumerate all token issuers and verification paths
- [ ] Enumerate all session cookie domains and settings
- [ ] Enumerate all OAuth providers and callback URLs
- [ ] Enumerate all auth-related env vars and secret stores

## B. Canonical Contract

- [ ] Single interactive auth hub URL selected
- [ ] Single token issuer selected
- [ ] Callback contract defined for each client type (web SPA, Next SSR, extension, CLI)
- [ ] Shared claims contract documented (`sub`, `role`, `client_id`, custom claims)

## C. Data Boundary Decisions

- [ ] Shared tables agreed (profiles, settings, subscriptions, shared webhook idempotency)
- [ ] App-local tables agreed (product domain data)
- [ ] Ownership and RLS policy boundaries documented

## D. Extension Strategy

- [ ] Explicit mode switch exists: `byok` vs `oauth`
- [ ] OAuth mode capability deltas documented in UI
- [ ] Token refresh path is deterministic and tested
- [ ] Extension callback redirect is strictly validated

## E. Security Hardening

- [ ] Asymmetric JWT signing configured where required
- [ ] OAuth client redirect URIs are exact and minimal
- [ ] General auth redirect allow-list is narrowed to known domains
- [ ] Open-redirect protections implemented for `next` params
- [ ] No secrets in source, logs, or MCP copy snippets
- [ ] Server-side auth uses `getClaims()` (not `getSession()` or unverified cookies)
- [ ] Edge Function "Verify JWT" disabled after asymmetric key migration (built-in uses old HS256 secret)
- [ ] SSR middleware skips `/auth/callback` to prevent cookie race with `exchangeCodeForSession()`
- [ ] `NEXT_PUBLIC_*` env vars confirmed set on all deploy targets (build-time inlined)

## E2. OAuth 2.1 Server (if Supabase is IdP)

- [ ] OAuth Server enabled in Supabase dashboard
- [ ] Authorization path configured and implemented (`/oauth/consent`)
- [ ] Consent UI renders client name, scopes, redirect URI
- [ ] Approve/deny decision endpoint handles both outcomes
- [ ] OAuth app redirect URIs are exact-match (separate from general allow-list)
- [ ] Dynamic client registration disabled unless explicitly needed
- [ ] `client_id` claim available for RLS policies if needed

## E3. Supabase Key Transition

- [ ] Code prefers `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY` with `ANON_KEY` fallback
- [ ] Both env var names set on all deploy targets during transition
- [ ] `.env.example` documents both key names
- [ ] Redeployed after env var changes (Next.js build-time inlining)

## F. Verification

- [ ] Endpoint sweep shows no active legacy auth routes
- [ ] Typecheck/build passes in each repo
- [ ] Auth smoke tests pass (login, callback, refresh, logout, guarded routes)
- [ ] Red-team rounds run and high/critical findings resolved

## G. Cleanup

- [ ] Legacy modules removed (not just deprecated)
- [ ] Legacy docs/examples updated or archived
- [ ] Roadmap updated with outcomes and deferred items
