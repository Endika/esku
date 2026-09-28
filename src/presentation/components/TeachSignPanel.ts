import type { LandmarkFrame } from '@domain/landmarks/value-objects/LandmarkFrame';
import type { CustomSign } from '@domain/recognition/entities/CustomSign';
import { MIN_PROTOTYPES_PER_SIGN } from '@domain/recognition/entities/CustomSign';

/**
 * The sign was saved or deleted, but the live recogniser still holds the old set until the app
 * reloads. Thrown by `save` and `remove` so the panel does not report a change that happened
 * as a failure.
 */
export class TaughtSignsNotReloadedError extends Error {
  constructor(cause: unknown) {
    super('The taught signs changed but the recogniser could not reload them', { cause });
    this.name = 'TaughtSignsNotReloadedError';
  }
}

const RECORDING_PROMPT = 'Haz el signo ahora. Para al terminar y se guarda solo.';

export interface TeachSignPanelPorts {
  /**
   * Resolves with the next completed sign performed in front of the camera, or rejects with an
   * error named `CaptureCancelledError` once `cancelCapture` or stopping the camera drops it.
   */
  captureWindow(): Promise<readonly LandmarkFrame[]>;
  cancelCapture(): void;
  isCameraRunning(): boolean;
  save(text: string, examples: readonly (readonly LandmarkFrame[])[]): Promise<void>;
  list(): Promise<CustomSign[]>;
  remove(id: string): Promise<void>;
}

/**
 * The teach-a-sign flow: name a word, perform it a few times, save.
 *
 * Recording reuses the live segmenter through `captureWindow`, so an example is delimited
 * exactly as a recognised sign will be — start moving, hold still to finish.
 */
export class TeachSignPanel {
  private takes: (readonly LandmarkFrame[])[] = [];
  private recording = false;

  constructor(
    private readonly root: HTMLElement,
    private readonly ports: TeachSignPanelPorts,
  ) {
    this.render();
    void this.refreshList();
  }

  private get input(): HTMLInputElement {
    return this.root.querySelector<HTMLInputElement>('#sign-text')!;
  }

  private render(): void {
    // Teaching and the resulting list are one subject, and an empty list did not deserve a
    // card of its own on a page that already ran three screens long.
    this.root.innerHTML = `
      <details class="card">
        <summary class="card__summary">
          <h2 class="card__title">Enseñar un signo</h2>
          <span class="card__note">Cualquier signo, cualquier lengua, sólo aquí</span>
        </summary>

        <div class="card__content">
          <p class="card__body">
            Graba el mismo signo ${MIN_PROTOTYPES_PER_SIGN} veces y escribe la palabra que debe
            aparecer.
          </p>

          <label class="field">
            <span class="field__label">Palabra que se escribirá</span>
            <input id="sign-text" class="field__input" type="text" maxlength="40"
                   placeholder="ibuprofeno" autocomplete="off" />
          </label>

          <div class="takes" id="takes"></div>

          <div class="actions">
            <button class="button" id="record" type="button">Grabar una toma</button>
            <button class="button button--quiet" id="save" type="button" disabled>Guardar</button>
            <button class="button button--quiet" id="reset" type="button">Descartar tomas</button>
          </div>

          <p class="status" id="teach-status" role="status"></p>

          <h3 class="card__subtitle">Tus signos</h3>
          <ul class="signs" id="signs"></ul>
        </div>
      </details>
    `;

    this.button('record').addEventListener('click', () => void this.recordTake());
    this.button('save').addEventListener('click', () => void this.save());
    this.button('reset').addEventListener('click', () => {
      this.ports.cancelCapture();
      this.recording = false;
      this.takes = [];
      this.renderTakes();
      this.say('Tomas descartadas.');
    });
  }

  private button(id: string): HTMLButtonElement {
    return this.root.querySelector<HTMLButtonElement>(`#${id}`)!;
  }

  private get status(): HTMLElement {
    return this.root.querySelector<HTMLElement>('#teach-status')!;
  }

  private say(message: string): void {
    this.status.textContent = message;
  }

  private async recordTake(): Promise<void> {
    if (this.recording) return;
    if (!this.ports.isCameraRunning()) {
      this.say('Enciende la cámara antes de grabar.');
      return;
    }

    this.recording = true;
    this.button('record').disabled = true;
    this.say(RECORDING_PROMPT);

    try {
      const window = await this.ports.captureWindow();
      this.takes.push(window);
      this.say(`Toma ${this.takes.length} guardada.`);
    } catch (error) {
      if (error instanceof Error && error.name === 'CaptureCancelledError') {
        // Discarding says so itself; stopping the camera would leave the prompt standing.
        if (this.status.textContent === RECORDING_PROMPT) this.say('Toma interrumpida.');
      } else {
        this.say('No se pudo grabar la toma.');
        console.error(error);
      }
    } finally {
      this.recording = false;
      this.button('record').disabled = false;
      this.renderTakes();
    }
  }

  private renderTakes(): void {
    const total = Math.max(MIN_PROTOTYPES_PER_SIGN, this.takes.length);
    const dots = Array.from({ length: total }, (_, i) =>
      i < this.takes.length
        ? '<span class="take take--done"></span>'
        : '<span class="take"></span>',
    ).join('');
    this.root.querySelector<HTMLElement>('#takes')!.innerHTML = dots;
    this.button('save').disabled = this.takes.length < MIN_PROTOTYPES_PER_SIGN;
  }

  private async save(): Promise<void> {
    let reloaded = true;
    try {
      await this.ports.save(this.input.value, this.takes);
    } catch (error) {
      if (isNotReloaded(error)) {
        reloaded = false;
        console.error(error);
      } else {
        // The domain errors carry the reason; showing it beats a generic failure.
        const reason = error instanceof Error ? reasonFor(error) : null;
        this.say(reason ?? 'No se pudo guardar.');
        if (reason === null) console.error(error);
        return;
      }
    }
    this.takes = [];
    this.input.value = '';
    this.renderTakes();
    const listed = await this.refreshList();
    if (!reloaded) this.say('Signo guardado, pero no se reconocerá hasta recargar la app.');
    else if (!listed) this.say('Signo guardado, pero no se pudo actualizar la lista.');
    else this.say('Signo guardado. Ya lo reconoce.');
  }

  /** Never throws: a list that fails to load says so in place and reports `false`. */
  private async refreshList(): Promise<boolean> {
    const list = this.root.querySelector<HTMLElement>('#signs')!;
    let signs: CustomSign[];
    try {
      signs = await this.ports.list();
    } catch (error) {
      list.innerHTML = '<li class="signs__empty">No se pudo cargar la lista de signos.</li>';
      console.error(error);
      return false;
    }

    if (signs.length === 0) {
      list.innerHTML = '<li class="signs__empty">Todavía no le has enseñado ninguno.</li>';
      return true;
    }

    list.innerHTML = signs
      .map(
        (sign) => `
          <li class="signs__item">
            <span>${escapeHtml(sign.text)}</span>
            <button class="button button--quiet button--small" data-delete="${escapeHtml(sign.id)}"
                    type="button">Borrar</button>
          </li>`,
      )
      .join('');

    list.querySelectorAll<HTMLButtonElement>('[data-delete]').forEach((button) => {
      button.addEventListener('click', async () => {
        let reloaded = true;
        try {
          await this.ports.remove(button.dataset.delete!);
        } catch (error) {
          console.error(error);
          if (!isNotReloaded(error)) {
            this.say('No se pudo borrar el signo.');
            return;
          }
          reloaded = false;
        }
        const listed = await this.refreshList();
        if (!reloaded)
          this.say('Signo borrado, pero se seguirá reconociendo hasta recargar la app.');
        else if (!listed) this.say('Signo borrado, pero no se pudo actualizar la lista.');
        else this.say('Signo borrado.');
      });
    });
    return true;
  }
}

function isNotReloaded(error: unknown): boolean {
  return error instanceof Error && error.name === 'TaughtSignsNotReloadedError';
}

function reasonFor(error: Error): string | null {
  switch (error.name) {
    case 'EmptySignTextError':
      return 'Escribe la palabra que debe aparecer.';
    case 'NotEnoughExamplesError':
      return `Hacen falta ${MIN_PROTOTYPES_PER_SIGN} tomas en las que se vea la mano.`;
    case 'DuplicateSignTextError':
      return 'Ya le has enseñado un signo con esa palabra.';
    default:
      return null;
  }
}

/** The word is user input rendered into innerHTML, so it must not be able to carry markup. */
function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
