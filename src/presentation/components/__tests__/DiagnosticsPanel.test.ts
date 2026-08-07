import {
  EMPTY_DIAGNOSTICS,
  type RecognitionDiagnostics,
} from '@domain/recognition/value-objects/RecognitionDiagnostics';
import { DiagnosticsPanel } from '@presentation/components/DiagnosticsPanel';
import { beforeEach, describe, expect, it } from 'vitest';

const reading: RecognitionDiagnostics = {
  ...EMPTY_DIAGNOSTICS,
  framesSeen: 42,
  windowsClosed: 3,
  windowsTooShort: 1,
};

/** The toggle event is queued, not synchronous — opening is observable on the next tick. */
const openAndSettle = async (details: HTMLDetailsElement): Promise<void> => {
  details.open = true;
  await new Promise((resolve) => setTimeout(resolve, 0));
};

describe('DiagnosticsPanel', () => {
  let root: HTMLElement;
  let panel: DiagnosticsPanel;

  beforeEach(() => {
    root = document.createElement('div');
    panel = new DiagnosticsPanel(root);
  });

  it('starts shut, so it costs one line on a page read by someone not debugging', () => {
    expect(root.querySelector<HTMLDetailsElement>('details')?.open).toBe(false);
    expect(root.querySelector('#diag-body')?.children).toHaveLength(0);
  });

  it('shows the last reading taken while it was shut, not an empty panel', async () => {
    panel.update(reading);
    await openAndSettle(root.querySelector<HTMLDetailsElement>('details')!);

    expect(root.querySelector('#diag-body')?.textContent).toContain('42');
    expect(root.querySelector('#diag-body')?.textContent).toContain('3 (1 descartadas por cortas)');
  });

  it('keeps updating while open, since it is read with the camera running', async () => {
    await openAndSettle(root.querySelector<HTMLDetailsElement>('details')!);
    panel.update({ ...reading, framesSeen: 99 });

    expect(root.querySelector('#diag-body')?.textContent).toContain('99');
  });
});
