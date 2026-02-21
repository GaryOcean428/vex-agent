import { Navigate, Outlet } from 'react-router-dom';
import { useAuth } from './useAuth.ts';

export default function ProtectedRoute() {
  const { authenticated, loading } = useAuth();

  if (loading) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        background: 'var(--bg)',
        color: 'var(--text-secondary)',
        fontFamily: 'var(--mono)',
        fontSize: '14px',
      }}>
        Authenticating...
      </div>
    );
  }

  if (!authenticated) {
    return <Navigate to="/login" replace />;
  }

  return <Outlet />;
}
