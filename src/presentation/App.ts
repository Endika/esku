import { Container } from '@bootstrap/Container';
import { CameraUnavailableError } from '@domain/landmarks/services/ILandmarkSource';
import { UNSUPPORTED_LETTERS } from '@infrastructure/recognition/packs/lseAlphabet';
import { DiagnosticsPanel } from '@presentation/components/DiagnosticsPanel';
import {
  LandmarkOverlay,
  type OverlayState,
  PART_COLOURS,
  PART_LABELS,
  PART_ORDER,
  type PartPresence,
} from '@presentation/components/LandmarkOverlay';
import { StoragePanel } from '@presentation/components/StoragePanel';
import { TeachSignPanel } from '@presentation/components/TeachSignPanel';

declare const __APP_VERSION__: string;

export function renderApp(root: HTMLElement): void {
  root.innerHTML = `
    <!--
      Everything needed to actually read signs lives in one viewport-tall column, so that
      turning the camera on never means scrolling to reach a control. Secondary panels stay
      below it rather than behind a sheet: the diagnostics panel in particular is read *while*
      the camera runs, and the sticky action bar keeps the controls reachable down there.
    -->
    <div class="shell" id="shell">
      <header class="masthead">
        <img class="masthead__mark" src="${import.meta.env.BASE_URL}favicon.svg" alt="" width="44" height="44" />
        <div>
          <h1 class="masthead__title">Esku</h1>
          <p class="masthead__tagline">Lengua de signos a texto, sin conexión</p>
        </div>
      </header>

      <div class="stage">
        <video id="video" class="stage__video" playsinline muted></video>
        <canvas id="overlay" class="stage__overlay" aria-hidden="true"></canvas>
        <p class="stage__placeholder" id="placeholder">
          La cámara se activa al empezar.<br />El vídeo no se graba ni sale del dispositivo.
        </p>
        <p class="stage__hint" id="hint" hidden></p>

        <ul class="parts" id="parts">
          ${PART_ORDER.map(
            (part) => `
            <li class="part" data-part="${part}">
              <span class="part__dot" style="--part: ${PART_COLOURS[part]}"></span>
              ${PART_LABELS[part]}
            </li>`,
          ).join('')}
        </ul>

        <!-- On the camera, not in the action bar: it is a control about the camera, the same
             argument that put the body-part chips here. Off, it would be a button that does
             nothing visible. -->
        <button class="stage__flip" id="flip" type="button" aria-label="Cambiar a cámara trasera">
          <svg viewBox="0 0 24 24" width="21" height="21" fill="none" stroke="currentColor"
               stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M4 9h3l1.6-2.2h6.8L17 9h3v10H4z" />
            <path d="M9.6 14.2a2.6 2.6 0 0 0 4.9.9M14.4 13a2.6 2.6 0 0 0-4.9-.9" />
            <path d="M9.4 10.6v1.5h1.5M14.6 15.6v-1.5h-1.5" />
          </svg>
        </button>
      </div>

      <div class="transcript" id="transcript" aria-live="polite"></div>

      <p class="status" id="status" role="status"></p>

      <!-- One row: the four buttons this replaces wrapped to two on a 390 px phone and to
           three at 320 px, and two of them acted on a transcript that did not exist yet. -->
      <div class="actions actions--bar" id="actions">
        <button class="button button--grow" id="toggle" type="button">Empezar a leer</button>
        <div class="actions__edit" id="edit" hidden>
          <button class="button button--quiet" id="undo" type="button">Deshacer</button>
          <button class="button button--quiet" id="clear" type="button">Limpiar</button>
        </div>
      </div>
    </div>

    <div id="teach"></div>
    <div id="storage"></div>
    <div id="diagnostics"></div>

    <!--
      The reliability figure lives in the summary, not behind it. Folding this card away
      saves 363 px, but what made it worth having was saying out loud that the model is
      fallible — so that sentence has to survive the fold.
    -->
    <details class="card">
      <summary class="card__summary">
        <h2 class="card__title">Qué reconoce</h2>
        <span class="card__note">238 signos LSE y el alfabeto — acierta 2 de cada 3 veces</span>
      </summary>

      <div class="card__content">
        <p class="card__body">
          <strong>Vocabulario LSE:</strong> 238 signos de ámbito sanitario, entrenados sobre
          SWL-LSE. Acierta el signo exacto en torno a <strong>2 de cada 3 veces</strong>, y está
          entre sus tres primeras opciones en <strong>8 de cada 10</strong>. Es un modelo real,
          no infalible: revisa el texto antes de darlo por bueno.
        </p>
        <p class="card__body" style="margin-top: 10px">
          <strong>Alfabeto dactilológico:</strong> para deletrear cualquier palabra fuera de ese
          vocabulario. Todavía no distingue <strong>${UNSUPPORTED_LETTERS.join(', ')}</strong>:
          unas se trazan con movimiento y otras dependen de la orientación de la palma.
        </p>
      </div>
    </details>

    <p class="footnote">
      v${__APP_VERSION__} · Vocabulario LSE sobre
      <a href="https://zenodo.org/records/13691887" rel="noreferrer">SWL-LSE</a> (CC-BY-4.0)
    </p>
  `;

  const video = must<HTMLVideoElement>(root, '#video');
  const overlayCanvas = must<HTMLCanvasElement>(root, '#overlay');
  const placeholder = must<HTMLElement>(root, '#placeholder');
  const hint = must<HTMLElement>(root, '#hint');
  const transcriptEl = must<HTMLElement>(root, '#transcript');
  const toggle = must<HTMLButtonElement>(root, '#toggle');
  const status = must<HTMLElement>(root, '#status');
  const edit = must<HTMLElement>(root, '#edit');

  // In camera mode the bar is fixed, so the page reserves its height at the bottom. Measured
  // rather than hard-coded: it grows a second row on a narrow phone once the transcript has
  // text, and a stale constant would leave the last panel hidden underneath it.
  const actions = must<HTMLElement>(root, '#actions');
  new ResizeObserver(() => {
    root.style.setProperty('--bar', `${actions.offsetHeight}px`);
  }).observe(actions);

  const parts = must<HTMLElement>(root, '#parts');
  const container = new Container(video);
  const { recognize } = container;
  const overlay = new LandmarkOverlay(overlayCanvas, video);
  const diagnostics = new DiagnosticsPanel(must<HTMLElement>(root, '#diagnostics'));

  /** Green when a part is being tracked, red when it is not — at a glance, per part. */
  const showPresence = (presence: PartPresence | null) => {
    for (const part of PART_ORDER) {
      const chip = parts.querySelector<HTMLElement>(`[data-part="${part}"]`);
      chip?.classList.toggle('part--on', presence?.[part] === true);
      chip?.classList.toggle('part--off', presence !== null && presence[part] === false);
    }
  };
  let running = false;

  const render = (text: string, candidates: readonly { gloss: { text: string } }[]) => {
    transcriptEl.textContent = text;
    // Undo and clear exist only once there is something to undo or clear. Every path that
    // changes the transcript comes through here, so this is the single place that decides.
    edit.hidden = text.length === 0;
    const top = candidates[0]?.gloss.text;
    hint.hidden = !top;
    if (top) hint.textContent = top.toUpperCase();
  };

  /**
   * Three states, because "nothing happened" has three different causes and the user can
   * only act on the right one: no hand in frame, hand tracked but shape unknown, or reading.
   */
  const describeTracking = (hands: number, hasCandidate: boolean): [OverlayState, string] => {
    if (hands === 0) return ['searching', 'Buscando la mano. Ponla dentro del encuadre.'];
    if (!hasCandidate) return ['tracking', 'Mano detectada. Esa forma todavía no la conozco.'];
    return ['recognised', 'Leyendo.'];
  };

  toggle.addEventListener('click', async () => {
    if (running) {
      recognize.stop();
      running = false;
      root.classList.remove('is-running');
      toggle.textContent = 'Empezar a leer';
      placeholder.hidden = false;
      hint.hidden = true;
      video.classList.remove('is-live');
      overlayCanvas.classList.remove('is-live');
      overlay.clear();
      showPresence(null);
      status.textContent = 'Cámara apagada.';
      return;
    }

    toggle.disabled = true;
    // The engine is ~29 MB on first run and cached after; say so rather than look frozen.
    status.textContent = 'Preparando el motor de reconocimiento…';
    try {
      // Both engines load before the camera opens, so the first sign is already recognisable
      // rather than silently ignored while weights are still arriving.
      await Promise.all([container.vocabulary.load(), container.taught.load()]);
      await recognize.start((update) => {
        const { transcript, candidates, frame } = update;
        render(transcript.toText(), candidates);
        const [state, message] = describeTracking(frame.hands.length, candidates.length > 0);
        showPresence(overlay.draw(frame, state));
        status.textContent = message;
        diagnostics.update(update.diagnostics);
      });
      running = true;
      // Camera mode: the masthead folds away, the video takes the height the fixed 3/4 ratio
      // used to claim regardless of device, and the controls pin to the bottom of the screen.
      root.classList.add('is-running');
      placeholder.hidden = true;
      video.classList.add('is-live');
      overlayCanvas.classList.add('is-live');
      toggle.textContent = 'Parar';
    } catch (error) {
      status.textContent =
        error instanceof CameraUnavailableError
          ? 'No hay cámara o se denegó el permiso. Revísalo en los ajustes del navegador.'
          : 'No se pudo iniciar el reconocimiento.';
      // The camera never opened, so the layout must not be left claiming it did.
      root.classList.remove('is-running');
      // Without the real cause in the console this is undiagnosable from a bug report.
      console.error(error);
    } finally {
      toggle.disabled = false;
    }
  });

  must<HTMLButtonElement>(root, '#undo').addEventListener('click', () => {
    recognize.undo();
    render(recognize.current.toText(), []);
  });

  must<HTMLButtonElement>(root, '#clear').addEventListener('click', () => {
    recognize.clear();
    render(recognize.current.toText(), []);
  });

  const flip = must<HTMLButtonElement>(root, '#flip');
  flip.addEventListener('click', async () => {
    const next = container.source.camera === 'user' ? 'environment' : 'user';
    flip.disabled = true;
    try {
      await container.source.useCamera(next);
      // Only the selfie view is mirrored. Un-mirroring the rear camera matters beyond looks:
      // the overlay is mirrored to match the video, so the two must agree or the skeleton
      // lands on the wrong side of the screen.
      const mirrored = next === 'user';
      video.classList.toggle('is-flipped', !mirrored);
      overlayCanvas.classList.toggle('is-flipped', !mirrored);
      // The icon says "switch", so the name has to say which way — it is the only label.
      flip.setAttribute(
        'aria-label',
        mirrored ? 'Cambiar a cámara trasera' : 'Cambiar a cámara frontal',
      );
      status.textContent = mirrored
        ? 'Cámara frontal: para signar tú.'
        : 'Cámara trasera: para leer a quien tienes delante.';
    } catch {
      status.textContent = 'No se pudo cambiar de cámara.';
    } finally {
      flip.disabled = false;
    }
  });

  new StoragePanel(must<HTMLElement>(root, '#storage'), {
    isSupported: () => container.engineStorage.isSupported(),
    report: () => container.engineStorage.report(),
    clear: () => container.engineStorage.clear(),
    // Loading through the real source downloads exactly the WASM variant this browser will
    // use, rather than guessing and fetching both the SIMD and no-SIMD builds.
    // Store the files ourselves first, then initialise. Letting MediaPipe fetch them and
    // hoping the service worker caught it left the runtime uncached and the app broken
    // offline — the models were saved and the WASM was not.
    preload: async (onProgress) => {
      await container.engineStorage.warm(onProgress);
      await Promise.all([container.source.load(), container.vocabulary.load()]);
    },
  });

  new TeachSignPanel(must<HTMLElement>(root, '#teach'), {
    captureWindow: () => recognize.captureWindow(),
    cancelCapture: () => {
      recognize.cancelCapture();
    },
    isCameraRunning: () => running,
    save: async (text, examples) => {
      await container.teach.execute(text, examples);
      // Reload prototypes so the sign is recognised immediately, not after a restart.
      await container.taught.refresh();
    },
    list: () => container.manageCustomSigns.list(),
    remove: async (id) => {
      await container.manageCustomSigns.delete(id);
      await container.taught.refresh();
    },
  });
}

function must<T extends Element>(root: ParentNode, selector: string): T {
  const found = root.querySelector<T>(selector);
  if (!found) throw new Error(`Missing element: ${selector}`);
  return found;
}
