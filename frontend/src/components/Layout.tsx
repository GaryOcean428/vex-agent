import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { useHealth } from '../hooks/index.ts';
import { CommandPalette } from './CommandPalette.tsx';
import { ThemeSwitcher } from './ThemeSwitcher.tsx';

const NAV_ITEMS = [
  { to: '/chat', label: 'Chat', icon: '◈' },
  { to: '/dashboard', label: 'Dashboard', icon: '▣' },
  { to: '/dashboard/graph', label: 'Graph', icon: '◎' },
  { to: '/dashboard/memory', label: 'Memory', icon: '◇' },
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
      {/* Skip-to-content link */}
      <a href="#main-content" className="skip-link">
        Skip to main content
      </a>

      {/* Desktop / Tablet: Left nav rail */}
      <aside
        className="nav-rail"
        aria-label="Application navigation"
        role="navigation"
      >
        <div className="rail-header">
          <span
            className={`pulse-dot ${health?.status === 'ok' ? 'alive' : 'degraded'}`}
            aria-hidden="true"
          />
          <span className="rail-logo">V</span>
        </div>

        <nav className="rail-nav" aria-label="Primary navigation">
          {NAV_ITEMS.map(item => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/dashboard'}
              className={({ isActive }) => `rail-item ${isActive ? 'active' : ''}`}
              title={item.label}
              aria-label={item.label}
            >
              <span className="rail-icon" aria-hidden="true">{item.icon}</span>
              <span className="rail-label">{item.label}</span>
            </NavLink>
          ))}
        </nav>

        {isDashboard && (
          <nav className="dashboard-tabs" aria-label="Dashboard sections">
            <div className="tabs-label" aria-hidden="true">Pages</div>
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

        <div className="rail-footer" aria-label="System status">
          <ThemeSwitcher />
          {health && (
            <span
              className={`rail-backend ${health.backend}`}
              title={`Backend: ${health.backend} | Uptime: ${formatUptime(health.uptime)} | Cycles: ${health.cycle_count}`}
              aria-label={`Backend: ${health.backend}`}
            >
              {health.backend === 'ollama' ? '●' : health.backend === 'external' ? '◐' : '○'}
            </span>
          )}
          {health && (
            <span className="rail-version" aria-label={`Version ${health.version ?? '2.2.0'}`}>
              v{health.version ?? '2.2.0'}
            </span>
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

      {/* Mobile: Bottom tab bar */}
      <nav className="bottom-bar" aria-label="Mobile navigation">
        {NAV_ITEMS.map(item => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/dashboard'}
            className={({ isActive }) => `bottom-tab ${isActive ? 'active' : ''}`}
            aria-label={item.label}
          >
            <span className="bottom-icon" aria-hidden="true">{item.icon}</span>
            <span className="bottom-label">{item.label}</span>
          </NavLink>
        ))}
      </nav>

      <CommandPalette />
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
