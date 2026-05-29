import { useCallback, useEffect, useRef, useState } from 'react';
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

// Mobile bottom bar includes a Metrics link (since the sidebar is hidden on mobile)
const MOBILE_NAV_ITEMS = [
  { to: '/chat', label: 'Chat', icon: '◈' },
  { to: '/metrics', label: 'Metrics', icon: '◉' },
  { to: '/dashboard', label: 'Dash', icon: '▣' },
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

const NAV_PINNED_KEY = 'vex-nav-pinned';

export default function Layout() {
  const { data: health } = useHealth();
  const location = useLocation();
  const isDashboard = location.pathname.startsWith('/dashboard');

  // --- Nav rail: click-to-toggle (desktop) ---
  const [navExpanded, setNavExpanded] = useState(() => {
    try { return localStorage.getItem(NAV_PINNED_KEY) === 'true'; } catch { return false; }
  });
  const [navPinned, setNavPinned] = useState(() => {
    try { return localStorage.getItem(NAV_PINNED_KEY) === 'true'; } catch { return false; }
  });
  // No hover expand — click only (toggle via pin button)
  const handleNavPointerEnter = useCallback(() => {}, []);
  const handleNavPointerLeave = useCallback(() => {}, []);

  const togglePin = useCallback(() => {
    setNavPinned(prev => {
      const next = !prev;
      try { localStorage.setItem(NAV_PINNED_KEY, String(next)); } catch { /* noop */ }
      setNavExpanded(next);
      return next;
    });
  }, []);

  // --- Mobile drawer ---
  const [drawerOpen, setDrawerOpen] = useState(false);
  const toggleDrawer = useCallback(() => setDrawerOpen(v => !v), []);
  const closeDrawer = useCallback(() => setDrawerOpen(false), []);
  const drawerRef = useRef<HTMLElement>(null);

  // Close drawer on Escape key
  useEffect(() => {
    if (!drawerOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { setDrawerOpen(false); }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [drawerOpen]);

  // Trap focus inside open drawer
  useEffect(() => {
    if (!drawerOpen || !drawerRef.current) return;
    const drawer = drawerRef.current;
    const focusable = drawer.querySelectorAll<HTMLElement>(
      'a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])'
    );
    if (focusable.length === 0) return;

    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    first.focus();

    const trap = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;
      if (e.shiftKey) {
        if (document.activeElement === first) { e.preventDefault(); last.focus(); }
      } else {
        if (document.activeElement === last) { e.preventDefault(); first.focus(); }
      }
    };
    drawer.addEventListener('keydown', trap);
    return () => drawer.removeEventListener('keydown', trap);
  }, [drawerOpen]);

  // Determine grid class
  const isExpanded = navExpanded || navPinned;
  const layoutClass = [
    'layout',
    isExpanded ? 'nav-expanded' : '',
    navPinned ? 'nav-pinned' : '',
  ].filter(Boolean).join(' ');

  // Shared nav content rendered in both desktop rail and mobile drawer
  const navContent = (
    <>
      <div className="rail-header">
        <span
          className={`pulse-dot ${health?.status === 'ok' ? 'alive' : 'degraded'}`}
          aria-hidden="true"
        />
        <span className="rail-logo">V</span>
        {isExpanded && (
          <button
            className="rail-pin-btn"
            onClick={togglePin}
            aria-label={navPinned ? 'Unpin sidebar' : 'Pin sidebar open'}
            title={navPinned ? 'Unpin sidebar' : 'Pin sidebar open'}
          >
            <span aria-hidden="true">{navPinned ? '◆' : '◇'}</span>
          </button>
        )}
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
    </>
  );

  return (
    <div className={layoutClass}>
      {/* Skip-to-content link */}
      <a href="#main-content" className="skip-link">
        Skip to main content
      </a>

      {/* Desktop: Left nav rail with hover-expand */}
      <aside
        className="nav-rail"
        aria-label="Application navigation"
        onPointerEnter={handleNavPointerEnter}
        onPointerLeave={handleNavPointerLeave}
      >
        {navContent}
      </aside>

      {/* Mobile drawer scrim */}
      {drawerOpen && (
        <div
          className="nav-drawer-scrim"
          onClick={() => setDrawerOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Mobile drawer — clicking any link inside closes it */}
      <aside
        ref={drawerRef}
        className={`nav-drawer ${drawerOpen ? 'open' : ''}`}
        aria-label="Mobile navigation"
        onClick={(e) => {
          // Close drawer when a link is clicked
          if ((e.target as HTMLElement).closest('a')) closeDrawer();
        }}
      >
        {navContent}
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
        <button
          className="bottom-tab"
          onClick={toggleDrawer}
          aria-label="Menu"
          aria-expanded={drawerOpen}
        >
          <span className="bottom-icon" aria-hidden="true">{drawerOpen ? '\u2715' : '\u2630'}</span>
          <span className="bottom-label">Menu</span>
        </button>
        {MOBILE_NAV_ITEMS.map(item => (
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
