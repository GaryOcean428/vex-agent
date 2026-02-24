import React, { useCallback, useEffect, useState } from 'react';
import { ThemeContext } from './ThemeContext';
import type { Theme } from './ThemeContext';

function resolveIsDark(selected: Theme): boolean {
  return (
    selected === 'dark' ||
    (selected === 'system' &&
      window.matchMedia('(prefers-color-scheme: dark)').matches)
  );
}

function applyClass(dark: boolean) {
  if (dark) {
    document.documentElement.classList.add('dark');
  } else {
    document.documentElement.classList.remove('dark');
  }
}

function readStoredTheme(): Theme {
  return (localStorage.getItem('theme') as Theme | null) || 'system';
}

// Apply class eagerly on first module load so there is no flash
applyClass(resolveIsDark(readStoredTheme()));

export const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [theme, setThemeState] = useState<Theme>(readStoredTheme);
  const [isDark, setIsDark] = useState(() => resolveIsDark(readStoredTheme()));

  const applyTheme = useCallback((selectedTheme: Theme) => {
    const dark = resolveIsDark(selectedTheme);
    applyClass(dark);
    setIsDark(dark);
  }, []);

  // Listen for system theme changes
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = () => applyTheme(theme);

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, [applyTheme, theme]);

  const setTheme = (newTheme: Theme) => {
    setThemeState(newTheme);
    localStorage.setItem('theme', newTheme);
    applyTheme(newTheme);
  };

  return (
    <ThemeContext.Provider value={{ theme, setTheme, isDark }}>
      {children}
    </ThemeContext.Provider>
  );
};
