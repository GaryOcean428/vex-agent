import {
  useCallback,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import { API } from "../config/api-routes.ts";
import { AuthContext } from "./authContext.ts";

/** Provides authentication state to the component tree. */
export function AuthProvider({ children }: { children: ReactNode }) {
  const [authenticated, setAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);

  const checkAuth = useCallback(async () => {
    try {
      const resp = await fetch(API.authCheck);
      setAuthenticated(resp.ok);
    } catch {
      setAuthenticated(false);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    checkAuth();
  }, [checkAuth]);

  return (
    <AuthContext.Provider value={{ authenticated, loading, checkAuth }}>
      {children}
    </AuthContext.Provider>
  );
}
