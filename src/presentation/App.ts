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
    </div>

    <ul class="parts" id="parts">
      ${PART_ORDER.map(
        (part) => `
        <li class="part" data-part="${part}">
          <span class="part__dot" style="--part: ${PART_COLOURS[part]}"></span>
          ${PART_LABELS[part]}
        </li>`,
      ).join('')}
    </ul>

    <div class="transcript" id="transcript" aria-live="polite"></div>

    <div class="actions">
      <button class="button" id="toggle" type="button">Empezar a leer</button>
      <button class="button button--quiet" id="undo" type="button">Borrar último</button>
      <button class="button button--quiet" id="clear" type="button">Limpiar</button>
      <button class="button button--quiet" id="flip" type="button">Cámara trasera</button>
    </div>

    <p class="status" id="status" role="status"></p>

    <div id="teach"></div>
    <div id="storage"></div>
    <div id="diagnostics"></div>

    <section class="card">
      <h2 class="card__title">Qué reconoce, y con qué fiabilidad</h2>
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
    </section>

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
      placeholder.hidden = true;
      video.classList.add('is-live');
      overlayCanvas.classList.add('is-live');
      toggle.textContent = 'Parar';
    } catch (error) {
      status.textContent =
        error instanceof CameraUnavailableError
          ? 'No hay cámara o se denegó el permiso. Revísalo en los ajustes del navegador.'
          : 'No se pudo iniciar el reconocimiento.';
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
      flip.textContent = mirrored ? 'Cámara trasera' : 'Cámara frontal';
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
