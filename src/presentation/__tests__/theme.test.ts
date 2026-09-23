import { beforeEach, describe, expect, it, vi } from 'vitest';

let systemIsLight = false;
vi.stubGlobal('matchMedia', () => ({
  get matches() {
    return systemIsLight;
  },
  addEventListener: () => {},
}));

const { applyThemePreference, nextThemePreference, readThemePreference } = await import(
  '@presentation/theme'
);

describe('theme', () => {
  beforeEach(() => {
    localStorage.clear();
    systemIsLight = false;
    document.head.innerHTML = `
      <meta name="theme-color" media="(prefers-color-scheme: light)" content="#ffffff" />
      <meta name="theme-color" media="(prefers-color-scheme: dark)" content="#000000" />`;
  });

  it('cycles system, light, dark and back to system', () => {
    expect(nextThemePreference('system')).toBe('light');
    expect(nextThemePreference('light')).toBe('dark');
    expect(nextThemePreference('dark')).toBe('system');
  });

  it('follows the system until one is pinned', () => {
    systemIsLight = true;
    expect(readThemePreference()).toBe('system');
    applyThemePreference('system');
    expect(document.documentElement.dataset.theme).toBe('light');
  });

  it('remembers a pinned scheme and paints the browser chrome with it', () => {
    systemIsLight = true;
    applyThemePreference('dark');

    expect(document.documentElement.dataset.theme).toBe('dark');
    expect(readThemePreference()).toBe('dark');
    const chrome = [...document.querySelectorAll<HTMLMetaElement>('meta[name="theme-color"]')];
    expect(chrome.map((meta) => meta.content)).toEqual(['#000000', '#000000']);
  });

  it('forgets the pin on going back to the system', () => {
    applyThemePreference('light');
    applyThemePreference('system');

    expect(readThemePreference()).toBe('system');
    const chrome = [...document.querySelectorAll<HTMLMetaElement>('meta[name="theme-color"]')];
    expect(chrome.map((meta) => meta.content)).toEqual(['#ffffff', '#000000']);
  });
});
