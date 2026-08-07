import { StoragePanel, type StoragePanelPorts } from '@presentation/components/StoragePanel';
import { describe, expect, it } from 'vitest';

const ports: StoragePanelPorts = {
  isSupported: () => true,
  report: async () => ({ cachedBytes: 0, entries: 0, hasRuntime: false }),
  clear: async () => true,
  preload: async () => {},
};

describe('StoragePanel', () => {
  it('folds away, since choosing to pre-download is a once-ever decision', () => {
    const root = document.createElement('div');
    new StoragePanel(root, ports);

    const card = root.querySelector<HTMLDetailsElement>('details.card');
    expect(card?.open).toBe(false);
    expect(card?.querySelector('#preload')).not.toBeNull();
    expect(card?.querySelector('#clear')).not.toBeNull();
  });

  it('keeps the size in the summary, where it is read without opening anything', () => {
    const root = document.createElement('div');
    new StoragePanel(root, ports);

    expect(root.querySelector('summary')?.textContent).toContain('19 MB');
  });
});
