import type { LandmarkFrame } from '@domain/landmarks/value-objects/LandmarkFrame';
import type { CustomSign } from '@domain/recognition/entities/CustomSign';
import {
  TaughtSignsNotReloadedError,
  TeachSignPanel,
  type TeachSignPanelPorts,
} from '@presentation/components/TeachSignPanel';
import { beforeEach, describe, expect, it } from 'vitest';

/** The real ports, backed by an array — the panel cannot tell the difference. */
class FakeTeaching implements TeachSignPanelPorts {
  readonly signs: CustomSign[] = [];
  cameraRunning = true;
  /** Set to make captures wait for a sign, as the real camera does, instead of resolving. */
  holdCaptures = false;
  listFails = false;
  removeFails = false;
  /** The change lands but the live recogniser cannot reload, as App's ports report it. */
  reloadFails = false;
  private pending: ((error: Error) => void) | null = null;

  captureWindow(): Promise<readonly LandmarkFrame[]> {
    if (!this.holdCaptures) return Promise.resolve([]);
    return new Promise((_, reject) => {
      this.pending = reject;
    });
  }
  cancelCapture(): void {
    const cancelled = new Error('cancelled');
    cancelled.name = 'CaptureCancelledError';
    this.pending?.(cancelled);
    this.pending = null;
  }
  isCameraRunning(): boolean {
    return this.cameraRunning;
  }
  async save(text: string): Promise<void> {
    this.signs.push({ id: `id-${this.signs.length}`, text, prototypes: [], createdAtMs: 0 });
    this.reload();
  }
  async list(): Promise<CustomSign[]> {
    if (this.listFails) throw new Error('IndexedDB read failed');
    return [...this.signs];
  }
  async remove(id: string): Promise<void> {
    if (this.removeFails) throw new Error('IndexedDB delete failed');
    const at = this.signs.findIndex((sign) => sign.id === id);
    if (at >= 0) this.signs.splice(at, 1);
    this.reload();
  }
  private reload(): void {
    if (this.reloadFails) throw new TaughtSignsNotReloadedError(new Error('refresh failed'));
  }
}

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

describe('TeachSignPanel', () => {
  let root: HTMLElement;
  let ports: FakeTeaching;

  beforeEach(() => {
    root = document.createElement('div');
    ports = new FakeTeaching();
    new TeachSignPanel(root, ports);
  });

  it('folds into one closed card, so it costs a line on the page and not a screen', () => {
    const cards = root.querySelectorAll('details.card');
    expect(cards).toHaveLength(1);
    expect((cards[0] as HTMLDetailsElement).open).toBe(false);
  });

  it('keeps the form and the list of taught signs together under that one card', () => {
    const content = root.querySelector('details.card .card__content')!;
    expect(content.querySelector('#sign-text')).not.toBeNull();
    expect(content.querySelector('#signs')).not.toBeNull();
  });

  it('names itself in the summary, which is all that is readable while shut', () => {
    expect(root.querySelector('summary')?.textContent).toContain('Enseñar un signo');
  });

  it('still lists a sign after saving it', async () => {
    await ports.save('ibuprofeno');
    new TeachSignPanel(root, ports);
    await Promise.resolve();

    expect(root.querySelector('#signs')?.textContent).toContain('ibuprofeno');
  });

  it('says the list is empty rather than showing nothing at all', async () => {
    await Promise.resolve();
    expect(root.querySelector('#signs')?.textContent).toContain('Todavía no');
  });

  describe('when something fails', () => {
    const click = (selector: string) => root.querySelector<HTMLButtonElement>(selector)!.click();
    const status = () => root.querySelector('#teach-status')?.textContent;

    async function recordThreeTakes() {
      for (let take = 0; take < 3; take += 1) {
        click('#record');
        await tick();
      }
    }

    it('frees the record button when the takes are discarded mid-recording', async () => {
      ports.holdCaptures = true;
      click('#record');
      expect(root.querySelector<HTMLButtonElement>('#record')!.disabled).toBe(true);

      click('#reset');
      await tick();

      expect(root.querySelector<HTMLButtonElement>('#record')!.disabled).toBe(false);
      expect(status()).toBe('Tomas descartadas.');
    });

    it('does not claim the save failed when only the list after it did', async () => {
      await recordThreeTakes();
      root.querySelector<HTMLInputElement>('#sign-text')!.value = 'ibuprofeno';
      ports.listFails = true;

      click('#save');
      await tick();

      expect(ports.signs.map((sign) => sign.text)).toEqual(['ibuprofeno']);
      expect(status()).not.toMatch(/No se pudo guardar/);
      expect(status()).toMatch(/Signo guardado/);
      expect(root.querySelector('#signs')?.textContent).toContain('No se pudo cargar');
    });

    it('says a saved sign needs a reload when the recogniser could not pick it up', async () => {
      await recordThreeTakes();
      root.querySelector<HTMLInputElement>('#sign-text')!.value = 'ibuprofeno';
      ports.reloadFails = true;

      click('#save');
      await tick();

      expect(status()).toBe('Signo guardado, pero no se reconocerá hasta recargar la app.');
      expect(root.querySelector('#signs')?.textContent).toContain('ibuprofeno');
    });

    it('says a deleted sign lingers until a reload when the recogniser could not drop it', async () => {
      await ports.save('ibuprofeno');
      new TeachSignPanel(root, ports);
      await tick();
      ports.reloadFails = true;

      click('[data-delete]');
      await tick();

      expect(status()).toBe('Signo borrado, pero se seguirá reconociendo hasta recargar la app.');
      expect(root.querySelector('#signs')?.textContent).not.toContain('ibuprofeno');
    });

    it('says the sign could not be deleted and keeps it listed', async () => {
      await ports.save('ibuprofeno');
      new TeachSignPanel(root, ports);
      await tick();
      ports.removeFails = true;

      click('[data-delete]');
      await tick();

      expect(status()).toBe('No se pudo borrar el signo.');
      expect(root.querySelector('#signs')?.textContent).toContain('ibuprofeno');
    });
  });
});
