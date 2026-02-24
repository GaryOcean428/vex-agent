import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { useHealth } from '../hooks/index.ts';
import { CommandPalette } from './CommandPalette.tsx';

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
    <div className="flex h-screen w-full overflow-hidden bg-light-bg-primary dark:bg-dark-bg-primary text-light-text-primary dark:text-dark-text-primary transition-theme">
      {/* Skip-to-content link */}
      <a href="#main-content" className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-neon-electric-blue focus:text-white focus:rounded-lg">
        Skip to main content
      </a>

      {/* Desktop / Tablet: Left nav rail */}
      <aside
        className="hidden md:flex flex-col w-16 lg:w-64 border-r border-light-border dark:border-dark-border bg-light-bg-secondary dark:bg-dark-bg-secondary transition-all duration-300 z-20"
        aria-label="Application navigation"
        role="navigation"
      >
        <div className="h-14 flex items-center justify-center lg:justify-start lg:px-6 border-b border-light-border dark:border-dark-border shrink-0">
          <span
            className={`w-2 h-2 rounded-full mr-0 lg:mr-3 ${health?.status === 'ok' ? 'bg-status-success shadow-[0_0_8px_rgba(0,184,148,0.6)]' : 'bg-status-warning shadow-[0_0_8px_rgba(253,203,110,0.6)]'}`}
            aria-hidden="true"
          />
          <span className="hidden lg:block font-mono font-bold tracking-wider text-neon-electric-blue">VEX</span>
          <span className="lg:hidden font-mono font-bold text-neon-electric-blue">V</span>
        </div>

        <nav className="flex-1 py-4 flex flex-col gap-2 px-2 lg:px-3 overflow-y-auto" aria-label="Primary navigation">
          {NAV_ITEMS.map(item => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/dashboard'}
              className={({ isActive }) => `flex items-center justify-center lg:justify-start h-10 lg:px-3 rounded-lg transition-all duration-200 ${isActive ? 'bg-neon-electric-blue/10 text-neon-electric-blue' : 'text-light-text-secondary dark:text-dark-text-secondary hover:bg-light-bg-tertiary dark:hover:bg-dark-bg-tertiary hover:text-light-text-primary dark:hover:text-dark-text-primary'}`}
              title={item.label}
              aria-label={item.label}
            >
              <span className="font-mono text-lg" aria-hidden="true">{item.icon}</span>
              <span className="hidden lg:block ml-3 text-sm font-medium">{item.label}</span>
            </NavLink>
          ))}

          {isDashboard && (
            <div className="mt-6 hidden lg:block">
              <div className="px-3 mb-2 text-xs font-mono text-light-text-quaternary dark:text-dark-text-quaternary uppercase tracking-wider" aria-hidden="true">Pages</div>
              <div className="flex flex-col gap-1">
                {DASHBOARD_TABS.map(tab => (
                  <NavLink
                    key={tab.to}
                    to={tab.to}
                    end={'end' in tab && tab.end}
                    className={({ isActive }) => `px-3 py-1.5 text-sm rounded-md transition-colors ${isActive ? 'bg-light-bg-tertiary dark:bg-dark-bg-tertiary text-light-text-primary dark:text-dark-text-primary font-medium' : 'text-light-text-secondary dark:text-dark-text-secondary hover:text-light-text-primary dark:hover:text-dark-text-primary'}`}
                  >
                    {tab.label}
                  </NavLink>
                ))}
              </div>
            </div>
          )}
        </nav>

        <div className="p-4 border-t border-light-border dark:border-dark-border flex flex-col items-center lg:items-start gap-1 text-xs font-mono text-light-text-quaternary dark:text-dark-text-quaternary" aria-label="System status">
          {health && (
            <span
              className={`cursor-help ${health.backend === 'ollama' ? 'text-neon-electric-blue' : health.backend === 'external' ? 'text-neon-electric-purple' : 'text-light-text-quaternary dark:text-dark-text-quaternary'}`}
              title={`Backend: ${health.backend} | Uptime: ${formatUptime(health.uptime)} | Cycles: ${health.cycle_count}`}
              aria-label={`Backend: ${health.backend}`}
            >
              {health.backend === 'ollama' ? '●' : health.backend === 'external' ? '◐' : '○'}
              <span className="hidden lg:inline ml-2">{health.backend}</span>
            </span>
          )}
          {health && (
            <span className="hidden lg:inline" aria-label={`Version ${health.version ?? '2.2.0'}`}>
              v{health.version ?? '2.2.0'}
            </span>
          )}
        </div>
      </aside>

      <main
        id="main-content"
        className="flex-1 flex flex-col min-w-0 h-full relative z-0"
        aria-label="Main content"
        tabIndex={-1}
      >
        <Outlet />
      </main>

      {/* Mobile: Bottom tab bar */}
      <nav className="md:hidden flex items-center justify-around h-[calc(56px+env(safe-area-inset-bottom))] pb-[env(safe-area-inset-bottom)] bg-light-bg-secondary dark:bg-dark-bg-secondary border-t border-light-border dark:border-dark-border z-20" aria-label="Mobile navigation">
        {NAV_ITEMS.map(item => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/dashboard'}
            className={({ isActive }) => `flex flex-col items-center justify-center w-full h-full gap-1 transition-colors ${isActive ? 'text-neon-electric-blue' : 'text-light-text-secondary dark:text-dark-text-secondary'}`}
            aria-label={item.label}
          >
            <span className="font-mono text-lg" aria-hidden="true">{item.icon}</span>
            <span className="text-[10px] font-medium">{item.label}</span>
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
