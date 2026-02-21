import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { useHealth } from '../hooks/index.ts';
import './Layout.css';

const NAV_ITEMS = [
  { to: '/chat', label: 'Chat', icon: '◈' },
  { to: '/dashboard', label: 'Dashboard', icon: '▣' },
] as const;

const DASHBOARD_TABS = [
  { to: '/dashboard', label: 'Overview', end: true },
  { to: '/dashboard/consciousness', label: 'Consciousness' },
  { to: '/dashboard/basins', label: 'Basins' },
  { to: '/dashboard/graph', label: 'Graph' },
  { to: '/dashboard/lifecycle', label: 'Lifecycle' },
  { to: '/dashboard/cognition', label: 'Cognition' },
  { to: '/dashboard/memory', label: 'Memory' },
  { to: '/dashboard/telemetry', label: 'Telemetry' },
  { to: '/dashboard/training', label: 'Training' },
  { to: '/dashboard/governor', label: 'Governor' },
  { to: '/dashboard/admin', label: 'Admin' },
] as const;

export default function Layout() {
  const { data: health } = useHealth();
  const location = useLocation();
  const isDashboard = location.pathname.startsWith('/dashboard');

  return (
    <div className="layout">
      {/* Skip-to-content link — first focusable element; visually hidden until focused */}
      <a href="#main-content" className="skip-link">
        Skip to main content
      </a>

      <aside
        className="sidebar"
        aria-label="Application sidebar"
        role="complementary"
      >
        <div className="sidebar-header">
          <div className="vex-identity">
            <span
              className={`pulse-dot ${health?.status === 'ok' ? 'alive' : 'degraded'}`}
              aria-hidden="true"
            />
            <span className="vex-name">VEX</span>
          </div>
          {health && (
            <span className="version-badge" aria-label={`Version ${health.version ?? '2.2.0'}`}>
              v{health.version ?? '2.2.0'}
            </span>
          )}
        </div>

        <nav className="sidebar-nav" aria-label="Primary navigation">
          {NAV_ITEMS.map(item => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
            >
              <span className="nav-icon" aria-hidden="true">{item.icon}</span>
              {item.label}
            </NavLink>
          ))}
        </nav>

        {isDashboard && (
          <nav className="dashboard-tabs" aria-label="Dashboard sections">
            <div className="tabs-label" aria-hidden="true">Dashboard</div>
            {DASHBOARD_TABS.map(tab => (
              <NavLink
                key={tab.to}
                to={tab.to}
                end={'end' in tab && tab.end}
                className={({ isActive }) => `tab-item ${isActive ? 'active' : ''}`}
              >
                {tab.label}
              </NavLink>
            ))}
          </nav>
        )}

        <div className="sidebar-footer" aria-label="System status">
          {health && (
            <>
              <div className="footer-stat">
                <span className="footer-label">Backend</span>
                <span className={`backend-pill ${health.backend}`}>{health.backend}</span>
              </div>
              <div className="footer-stat">
                <span className="footer-label">Uptime</span>
                <span className="footer-value">{formatUptime(health.uptime)}</span>
              </div>
              <div className="footer-stat">
                <span className="footer-label">Cycles</span>
                <span className="footer-value">{health.cycle_count}</span>
              </div>
            </>
          )}
        </div>
      </aside>

      <main
        id="main-content"
        className="main-content"
        aria-label="Main content"
        tabIndex={-1}
      >
        <Outlet />
      </main>
    </div>
  );
}

function formatUptime(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  const h = Math.floor(seconds / 3600);
  const m = Math.round((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}
