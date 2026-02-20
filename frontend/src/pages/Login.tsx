import { useState, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext.tsx';
import { API } from '../config/api-routes.ts';
import './Login.css';

export default function Login() {
  const [token, setToken] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const { checkAuth } = useAuth();

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!token.trim()) return;
    setLoading(true);
    setError('');

    try {
      const resp = await fetch(API.chatAuth, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: token.trim() }),
      });

      if (resp.ok) {
        await checkAuth();
        navigate('/chat', { replace: true });
      } else {
        const data = await resp.json().catch(() => ({}));
        setError(data.error || 'Invalid token');
      }
    } catch {
      setError('Connection error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-page">
      <div className="login-container">
        <div className="login-header">
          <div className="login-pulse" />
          <h1 className="login-title">VEX</h1>
          <p className="login-subtitle">Consciousness Engine v2.2</p>
        </div>

        <form onSubmit={handleSubmit} className="login-form">
          <input
            type="password"
            value={token}
            onChange={e => setToken(e.target.value)}
            placeholder="Access token"
            autoComplete="off"
            autoFocus
            className="login-input"
          />
          <button
            type="submit"
            disabled={loading || !token.trim()}
            className="login-btn"
          >
            {loading ? 'Authenticating...' : 'Authenticate'}
          </button>
          {error && <div className="login-error">{error}</div>}
        </form>
      </div>
    </div>
  );
}
