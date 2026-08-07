import type { LandmarkFrame } from '@domain/landmarks/value-objects/LandmarkFrame';
import type { CustomSign } from '@domain/recognition/entities/CustomSign';
import { TeachSignPanel, type TeachSignPanelPorts } from '@presentation/components/TeachSignPanel';
import { beforeEach, describe, expect, it } from 'vitest';

/** The real ports, backed by an array — the panel cannot tell the difference. */
class FakeTeaching implements TeachSignPanelPorts {
  readonly signs: CustomSign[] = [];
  cameraRunning = true;

  captureWindow(): Promise<readonly LandmarkFrame[]> {
    return Promise.resolve([]);
  }
  cancelCapture(): void {}
  isCameraRunning(): boolean {
    return this.cameraRunning;
  }
  async save(text: string): Promise<void> {
    this.signs.push({ id: `id-${this.signs.length}`, text, prototypes: [], createdAtMs: 0 });
  }
  async list(): Promise<CustomSign[]> {
    return [...this.signs];
  }
  async remove(id: string): Promise<void> {
    const at = this.signs.findIndex((sign) => sign.id === id);
    if (at >= 0) this.signs.splice(at, 1);
  }
}

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
});
