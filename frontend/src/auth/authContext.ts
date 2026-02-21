/**
 * Auth context definition — types and context object only.
 * Separated so both AuthContext.tsx (provider) and useAuth.ts (hook)
 * can import without circular dependencies or fast-refresh violations.
 */
import { createContext } from "react";

export interface AuthState {
  authenticated: boolean;
  loading: boolean;
  checkAuth: () => Promise<void>;
}

export const AuthContext = createContext<AuthState>({
  authenticated: false,
  loading: true,
  checkAuth: async () => {},
});
