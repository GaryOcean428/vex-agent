#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
INCLUDE_ARCHIVE="${AUTH_SCAN_INCLUDE_ARCHIVE:-0}"
INCLUDE_DOCS="${AUTH_SCAN_INCLUDE_DOCS:-0}"

if ! command -v rg >/dev/null 2>&1; then
  echo "ERROR: ripgrep (rg) is required."
  exit 1
fi

if [[ ! -d "$ROOT" ]]; then
  echo "ERROR: root path not found: $ROOT"
  exit 1
fi

ROOT="$(cd "$ROOT" && pwd)"

echo "=== Auth Surface Scan ==="
echo "Root: $ROOT"
echo

scan() {
  local label="$1"
  local pattern="$2"
  local exit_code=0

  echo "--- $label ---"
  local -a globs=(
    '--glob' '!.git'
    '--glob' '!node_modules/**'
    '--glob' '!dist/**'
    '--glob' '!.next/**'
    '--glob' '!coverage/**'
    '--glob' '!**/build/**'
    '--glob' '!**/*.backup'
    '--glob' '!**/*.bak'
  )

  if [[ "$INCLUDE_ARCHIVE" != "1" ]]; then
    globs+=('--glob' '!docs/archive/**' '--glob' '!scripts/archive/**' '--glob' '!**/_legacy/**')
  fi
  if [[ "$INCLUDE_DOCS" != "1" ]]; then
    globs+=('--glob' '!docs/**')
  fi

  pushd "$ROOT" >/dev/null
  if rg -n --hidden "${globs[@]}" "$pattern" .; then
    exit_code=0
  else
    exit_code=$?
    if [[ $exit_code -eq 1 ]]; then
      echo "(no matches)"
      exit_code=0
    else
      popd >/dev/null
      return $exit_code
    fi
  fi
  popd >/dev/null
  echo
}

scan "Legacy auth route patterns" '/api/v1/auth/login|/api/v1/auth/signup|/api/v1/auth/refresh|/v1/auth/login|/v1/auth/signup|/v1/auth/refresh'
scan "Weak redirect patterns" '\*\.up\.railway\.app/\*\*|\*\.vercel\.app/\*\*|redirect_url|redirectTo|callback_url|callbackPath|additional_redirect_urls'
scan "JWT algorithm and key smells" 'HS256|JWT_SECRET|SUPABASE_SERVICE_ROLE_KEY|SUPABASE_SECRET_KEY|SUPABASE_ANON_KEY|SUPABASE_PUBLISHABLE_KEY'
scan "Extension auth mode and token storage references" 'chrome\.identity|launchWebAuthFlow|auth_token|refresh_token|supabase_url|supabase_anon_key|extensionAuthMode|byok'
scan "OAuth 2.1 Server routes and SDK usage" 'oauth/consent|oauth/decision|approveAuthorization|denyAuthorization|getAuthorizationDetails|client_id.*claim'
scan "Auth verification methods (getClaims vs getUser vs getSession)" 'supabase\.auth\.getClaims|supabase\.auth\.getUser|supabase\.auth\.getSession'
scan "Cookie domain configuration" 'COOKIE_DOMAIN|cookieDomain|cookie_domain|cookieOptions.*domain'

echo "Scan complete."
