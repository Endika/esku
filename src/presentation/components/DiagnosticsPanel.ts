import {
  EMPTY_DIAGNOSTICS,
  type RecognitionDiagnostics,
  type WindowVeto,
} from '@domain/recognition/value-objects/RecognitionDiagnostics';

/**
 * Shows what the pipeline did, so "it writes nothing" becomes a specific failure.
 *
 * Ships in the production build on purpose. The app is tested on a phone against the
 * deployed page, so a dev-only panel would never once be looked at while the bug is
 * reproducing.
 */
const VETO_LABELS: Record<WindowVeto, string> = {
  classifier: 'el modelo: ninguna opción llegó a su mínimo',
  stabilizer: 'el estabilizador: el modelo respondió pero se quedó corto',
  duplicate: 'repetido: mismo signo que el anterior',
};

export class DiagnosticsPanel {
  private open = false;
  private latest: RecognitionDiagnostics = EMPTY_DIAGNOSTICS;

  constructor(private readonly root: HTMLElement) {
    this.render();
  }

  update(diagnostics: RecognitionDiagnostics): void {
    this.latest = diagnostics;
    if (this.open) this.paint();
  }

  private render(): void {
    this.root.innerHTML = `
      <section class="card">
        <h2 class="card__title">Diagnóstico</h2>
        <p class="card__body">
          Para entender por qué no aparece una palabra. No hace falta para usar la app.
        </p>
        <div class="actions">
          <button class="button button--quiet" id="diag-toggle" type="button">Ver diagnóstico</button>
        </div>
        <dl class="diagnostics" id="diag-body" hidden></dl>
      </section>
    `;

    this.root.querySelector<HTMLButtonElement>('#diag-toggle')!.addEventListener('click', () => {
      this.open = !this.open;
      this.body().hidden = !this.open;
      this.root.querySelector<HTMLButtonElement>('#diag-toggle')!.textContent = this.open
        ? 'Ocultar diagnóstico'
        : 'Ver diagnóstico';
      if (this.open) this.paint();
    });
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
      ['Segmentador', d.segmenterActive ? `activo, ${d.pendingFrames} fotogramas` : 'en reposo'],
      ['Ventanas cerradas', `${d.windowsClosed} (${d.windowsTooShort} descartadas por cortas)`],
      ['Última ventana', d.lastWindowFrames ? `${d.lastWindowFrames} fotogramas` : '—'],
      ['Motor cargado', d.vocabularyReady ? 'sí' : 'no'],
      ['Veces consultado', `${d.vocabularyInvocations}`],
      ['Palabras escritas', `${d.wordsEmitted}`],
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
    `;
  }
}
