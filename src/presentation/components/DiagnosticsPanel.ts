import type { FrameCost } from '@domain/landmarks/value-objects/FrameCost';
import {
  EMPTY_DIAGNOSTICS,
  type RecognitionDiagnostics,
  type SignatureBlock,
  type WindowVeto,
} from '@domain/recognition/value-objects/RecognitionDiagnostics';

/**
 * The same statistics over SWL-LSE's test split — `tools/train`, 598 recordings.
 *
 * Medians, not means. The mean of a hand block is 3.37 and its median 0.73: three of the
 * sixty-nine floats are the wrist relative to the torso, and on the 0.01% of frames where
 * MediaPipe collapses shoulder width to nearly zero that division reaches six figures. A
 * mean reference flagged a perfectly healthy vector as broken by a factor of three.
 */
const EXPECTED: Record<string, { empty: number; magnitude: number }> = {
  'Mano derecha': { empty: 0.238, magnitude: 0.73 },
  'Mano izquierda': { empty: 0.377, magnitude: 0.73 },
  Torso: { empty: 0.0, magnitude: 0.51 },
  Cara: { empty: 0.002, magnitude: 0.15 },
};

const VETO_LABELS: Record<WindowVeto, string> = {
  classifier: 'el modelo: ninguna opción llegó a su mínimo',
  stabilizer: 'el estabilizador: el modelo respondió pero se quedó corto',
  duplicate: 'repetido: mismo signo que el anterior',
};

function frameCostLabel(cost: FrameCost | null): string {
  if (!cost) return '—';
  const total = cost.handsMs + cost.poseMs + cost.faceMs;
  const share = (ms: number) => `${ms.toFixed(0)} ms`;
  return (
    `${share(total)} · manos ${share(cost.handsMs)} · pose ${share(cost.poseMs)}` +
    ` · cara ${share(cost.faceMs)}`
  );
}

/**
 * Shows what the pipeline did, so "it writes nothing" becomes a specific failure.
 *
 * Ships in the production build on purpose. The app is tested on a phone against the
 * deployed page, so a dev-only panel would never once be looked at while the bug is
 * reproducing.
 */
export class DiagnosticsPanel {
  private latest: RecognitionDiagnostics = EMPTY_DIAGNOSTICS;

  constructor(private readonly root: HTMLElement) {
    this.render();
  }

  update(diagnostics: RecognitionDiagnostics): void {
    this.latest = diagnostics;
    // Painting a shut panel is work nobody sees, at the frame rate of the camera.
    if (this.details().open) this.paint();
  }

  private render(): void {
    this.root.innerHTML = `
      <details class="card">
        <summary class="card__summary">
          <h2 class="card__title">Diagnóstico</h2>
          <span class="card__note">Por qué no aparece una palabra</span>
        </summary>
        <dl class="diagnostics card__content" id="diag-body"></dl>
      </details>
    `;

    // Opening it must show the latest reading, not wait for the next frame — the panel is
    // also opened with the camera off, when no further update is ever coming.
    this.details().addEventListener('toggle', () => {
      if (this.details().open) this.paint();
    });
  }

  private details(): HTMLDetailsElement {
    return this.root.querySelector<HTMLDetailsElement>('details')!;
  }

  private body(): HTMLElement {
    return this.root.querySelector<HTMLElement>('#diag-body')!;
  }

  private paint(): void {
    const d = this.latest;
    // The three rival diagnoses, in the order you have to rule them out: is the segmenter
    // closing windows at all, is the engine being asked, and what did it actually score.
    const rows: [string, string][] = [
      ['Fotogramas', `${d.framesSeen} (${d.framesWithHands} con mano)`],
      ['Coste por fotograma', frameCostLabel(d.frameCost)],
      ['Segmentador', d.segmenterActive ? `activo, ${d.pendingFrames} fotogramas` : 'en reposo'],
      ['Ventanas cerradas', `${d.windowsClosed} (${d.windowsTooShort} descartadas por cortas)`],
      ['Última ventana', d.lastWindowFrames ? `${d.lastWindowFrames} fotogramas` : '—'],
      ['Motor cargado', d.vocabularyReady ? 'sí' : 'no'],
      ['Veces consultado', `${d.vocabularyInvocations}`],
      ['Palabras del vocabulario', `${d.wordsEmitted}`],
      ['Letras deletreadas', `${d.lettersEmitted}`],
      ['Bloqueado por', d.lastVeto ? VETO_LABELS[d.lastVeto] : '—'],
    ];

    const scores = d.lastRawTop.length
      ? d.lastRawTop
          .map((score) => `${score.text} ${(score.confidence * 100).toFixed(1)}%`)
          .join(' · ')
      : 'todavía nada';

    this.body().innerHTML = `
      ${rows
        .map(
          ([label, value]) =>
            `<div class="diagnostics__row"><dt>${label}</dt><dd>${value}</dd></div>`,
        )
        .join('')}
      <div class="diagnostics__row diagnostics__row--wide">
        <dt>Mejores opciones, sin filtrar</dt>
        <dd>${scores}</dd>
      </div>
      <div class="diagnostics__row diagnostics__row--wide">
        <dt>Lo que recibió el modelo (esperado entre paréntesis)</dt>
        <dd>${this.describeSignature()}</dd>
      </div>
    `;
  }

  /** Live feature magnitudes next to the training reference, one line per body part. */
  private describeSignature(): string {
    const profile = this.latest.lastSignature;
    if (!profile) return 'todavía nada';

    const parts: [string, SignatureBlock][] = [
      ['Mano derecha', profile.rightHand],
      ['Mano izquierda', profile.leftHand],
      ['Torso', profile.torso],
      ['Cara', profile.face],
    ];

    return parts
      .map(([label, block]) => {
        const reference = EXPECTED[label]!;
        const empty = `${(block.emptyFrames * 100).toFixed(0)}% vacío (${(reference.empty * 100).toFixed(0)}%)`;
        const size = `${block.meanMagnitude.toFixed(2)} (${reference.magnitude.toFixed(2)})`;
        // Flagged rather than left to be eyeballed: an order of magnitude is the signal.
        const off =
          block.emptyFrames - reference.empty > 0.3 ||
          block.meanMagnitude > reference.magnitude * 3 ||
          block.meanMagnitude < reference.magnitude / 3;
        return `<div${off ? ' class="diagnostics__off"' : ''}>${label}: ${empty} · ${size}</div>`;
      })
      .join('');
  }
}
