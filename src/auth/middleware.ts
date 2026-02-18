/**
 * Shared Auth Middleware — Session-based token authentication.
 *
 * Extracted from chat/router.ts so it can be applied globally in index.ts.
 * When CHAT_AUTH_TOKEN is set, all routes (except /health, /chat/auth, and
 * static assets) require a valid session cookie.
 */

import { Request, Response, NextFunction } from 'express';
import * as crypto from 'crypto';
import { config } from '../config';

/** Session cookie name */
export const SESSION_COOKIE = 'vex_session';

/** Session duration: 7 days in ms */
export const SESSION_TTL_MS = 7 * 24 * 60 * 60 * 1000;

interface Session {
  id: string;
  createdAt: number;
  expiresAt: number;
}

const sessions = new Map<string, Session>();

/** Paths that never require authentication */
const AUTH_EXEMPT_PATHS = new Set(['/health', '/chat/auth', '/login']);

/** Check if a path looks like a static asset (has a file extension) */
function isStaticAsset(path: string): boolean {
  return /\.\w{2,5}$/.test(path);
}

export function createSession(): string {
  const id = crypto.randomBytes(32).toString('hex');
  const now = Date.now();
  sessions.set(id, { id, createdAt: now, expiresAt: now + SESSION_TTL_MS });
  return id;
}

export function isValidSession(sessionId: string | undefined): boolean {
  if (!sessionId) return false;
  const session = sessions.get(sessionId);
  if (!session) return false;
  if (Date.now() > session.expiresAt) {
    sessions.delete(sessionId);
    return false;
  }
  return true;
}

export function getCookie(req: Request, name: string): string | undefined {
  const header = req.headers.cookie;
  if (!header) return undefined;
  const match = header.split(';').find((c) => c.trim().startsWith(name + '='));
  if (!match) return undefined;
  return match.split('=').slice(1).join('=').trim();
}

/**
 * Global auth middleware.
 *
 * - No-op when CHAT_AUTH_TOKEN is empty (dev mode).
 * - Always allows /health, /chat/auth, and static asset requests.
 * - Browser navigation requests (HTML accept) → redirect to /login.
 * - API requests → 401 JSON.
 */
export function requireAuth(req: Request, res: Response, next: NextFunction): void {
  // No auth configured — pass everything through
  if (!config.chatAuthToken) {
    next();
    return;
  }

  // Exempt paths that must always be accessible
  if (AUTH_EXEMPT_PATHS.has(req.path)) {
    next();
    return;
  }

  // Static assets (JS, CSS, images) must load for the login page to render
  if (isStaticAsset(req.path)) {
    next();
    return;
  }

  // Valid session cookie → authorized
  const sessionId = getCookie(req, SESSION_COOKIE);
  if (isValidSession(sessionId)) {
    next();
    return;
  }

  // Not authenticated — check if browser or API request
  const acceptsHtml = req.headers.accept?.includes('text/html');
  if (acceptsHtml) {
    res.redirect('/login');
    return;
  }

  res.status(401).json({ error: 'Authentication required' });
}
