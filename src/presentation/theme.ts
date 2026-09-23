/**
 * The colour scheme: the system's until the footnote's toggle pins one. index.html applies the
 * stored choice before first paint; this keeps it applied and lets the toggle change it.
 */
export type ThemePreference = 'system' | 'light' | 'dark';

const KEY = 'esku:theme';
const NEXT: Record<ThemePreference, ThemePreference> = {
  system: 'light',
  light: 'dark',
  dark: 'system',
};
/* The browser chrome matches the viewfinder's ground. */
const CHROME = { light: '#ffffff', dark: '#000000' };

const systemLight = window.matchMedia('(prefers-color-scheme: light)');

export function readThemePreference(): ThemePreference {
  try {
    const stored = localStorage.getItem(KEY);
    if (stored === 'light' || stored === 'dark') return stored;
  } catch {
    // Storage blocked: the system's scheme still works.
  }
  return 'system';
}

export function nextThemePreference(preference: ThemePreference): ThemePreference {
  return NEXT[preference];
}

export function applyThemePreference(preference: ThemePreference): void {
  const theme = preference === 'system' ? (systemLight.matches ? 'light' : 'dark') : preference;
  document.documentElement.dataset.theme = theme;
  for (const meta of document.querySelectorAll<HTMLMetaElement>('meta[name="theme-color"]')) {
    const own = meta.getAttribute('media')?.includes('light') ? 'light' : 'dark';
    meta.content = CHROME[preference === 'system' ? own : theme];
  }
  try {
    if (preference === 'system') localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, preference);
  } catch {
    // Not remembered, but applied for this visit.
  }
}

export function followSystemTheme(current: () => ThemePreference): void {
  systemLight.addEventListener('change', () => {
    if (current() === 'system') applyThemePreference('system');
  });
}
