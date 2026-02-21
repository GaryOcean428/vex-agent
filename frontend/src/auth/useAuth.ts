import { useContext } from "react";
import { AuthContext } from "./authContext.ts";

/** Returns the current authentication state from AuthProvider. */
export const useAuth = () => useContext(AuthContext);
