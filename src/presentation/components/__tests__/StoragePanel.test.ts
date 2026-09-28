import { ModelFileError } from '@infrastructure/recognition/modelFiles';
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

  describe('when the download fails', () => {
    const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

    async function preloadFailingWith(error: Error): Promise<string> {
      const root = document.createElement('div');
      new StoragePanel(root, {
        ...ports,
        preload: async () => {
          throw error;
        },
      });
      root.querySelector<HTMLButtonElement>('#preload')!.click();
      await tick();
      return root.querySelector('#storage-status')?.textContent ?? '';
    }

    it('blames the connection when the files did not arrive', async () => {
      const status = await preloadFailingWith(new Error('Could not cache 2 of 7 engine files'));

      expect(status).toMatch(/Comprueba la conexión/);
    });

    it('says the model is broken when the files arrived damaged', async () => {
      const status = await preloadFailingWith(new ModelFileError('lse-vocabulary.bin is short'));

      expect(status).not.toMatch(/conexión/);
      expect(status).toMatch(/Liberar espacio/);
    });
  });
});
