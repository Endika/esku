import { Container } from '@bootstrap/Container';
import { CameraUnavailableError } from '@domain/landmarks/services/ILandmarkSource';
import { WEAK_LETTERS } from '@infrastructure/recognition/CtcAlphabetClassifier';
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
import {
  applyThemePreference,
  followSystemTheme,
  nextThemePreference,
  readThemePreference,
  type ThemePreference,
} from '@presentation/theme';

declare const __APP_VERSION__: string;

export function renderApp(root: HTMLElement): void {
  root.innerHTML = `
    <!--
      The viewfinder owns the screen: everything needed to read signs is on it or under the
      video, so turning the camera on never means scrolling to reach a control. The tools
      slide up over it as a sheet on a phone and sit beside it on a wide screen, so the
      diagnostics panel can be read *while* the camera runs.
    -->
    <div class="viewer">
      <section class="viewfinder" id="viewfinder" aria-label="Cámara">
        <div class="hud">
          <p class="hud__status" id="status" role="status"></p>
          <!-- Up here with the camera's state, not in the bottom bar: it is a control about the
               camera. Off, it would be a button that does nothing visible. -->
          <button class="hud__flip" id="flip" type="button" aria-label="Cambiar a cámara trasera">
            <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor"
                 stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M4 9h3l1.6-2.2h6.8L17 9h3v10H4z" />
              <path d="M9.6 14.2a2.6 2.6 0 0 0 4.9.9M14.4 13a2.6 2.6 0 0 0-4.9-.9" />
              <path d="M9.4 10.6v1.5h1.5M14.6 15.6v-1.5h-1.5" />
            </svg>
          </button>
          <ul class="parts" id="parts" aria-label="Qué está viendo la cámara">
            ${PART_ORDER.map(
              (part) => `
              <li class="part" data-part="${part}" style="--part: ${PART_COLOURS[part]}">
                <svg class="part__mark" viewBox="0 0 12 12" width="12" height="12" aria-hidden="true">
                  <circle class="part__dot" cx="6" cy="6" r="3" />
                  <path class="part__tick" d="M2.5 6.3 5 8.6l4.5-5" />
                  <path class="part__cross" d="M3 3l6 6M9 3 3 9" />
                </svg>
                <span>${PART_LABELS[part]}</span>
                <span class="part__state"></span>
              </li>`,
            ).join('')}
          </ul>
        </div>

        <div class="frame" id="frame" data-track="idle">
          <video id="video" class="frame__video" playsinline muted></video>
          <canvas id="overlay" class="frame__overlay" aria-hidden="true"></canvas>

          <!-- The framing brackets are the tracking state: dim, white on a hand, yellow once
               hands, face and torso are all in frame. -->
          <span class="guide guide--tl" aria-hidden="true"></span>
          <span class="guide guide--tr" aria-hidden="true"></span>
          <span class="guide guide--bl" aria-hidden="true"></span>
          <span class="guide guide--br" aria-hidden="true"></span>

          <div class="idle" id="placeholder">
            <img class="idle__mark" src="${import.meta.env.BASE_URL}favicon.svg" alt="" width="56" height="56" />
            <h1 class="idle__title">Esku</h1>
            <p class="idle__tagline">Signos de LSE a texto, de uno en uno y sin conexión</p>
            <p class="idle__note">
              La cámara se activa al empezar. El vídeo no se graba ni sale del dispositivo.
            </p>
            <p class="idle__note">286 signos LSE y el alfabeto — acierta 2 de cada 3 veces</p>
          </div>

          <p class="hint" id="hint" hidden></p>
        </div>

        <div class="caption">
          <div class="transcript" id="transcript" aria-live="polite"></div>
          <!-- Undo and clear exist only once there is something to undo or clear. -->
          <div class="caption__edit" id="edit" hidden>
            <button class="chip-button" id="transcript-undo" type="button">
              <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"
                   stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M9 14 4 9l5-5" /><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11" />
              </svg>
              Deshacer
            </button>
            <button class="chip-button" id="transcript-clear" type="button">
              <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"
                   stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3" />
              </svg>
              Limpiar
            </button>
          </div>
        </div>

        <nav class="controls" aria-label="Controles">
          <button class="controls__tools" id="tools-open" type="button"
                  aria-controls="tools" aria-expanded="false">
            <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor"
                 stroke-width="1.8" stroke-linecap="round" aria-hidden="true">
              <path d="M4 7h10M18 7h2M4 17h2M10 17h10" /><circle cx="16" cy="7" r="2" /><circle cx="8" cy="17" r="2" />
            </svg>
            <span>Herramientas</span>
          </button>
          <button class="disc" id="toggle" type="button">
            <span class="disc__face" aria-hidden="true"></span>
            <span class="disc__label" id="toggle-label">Empezar a leer</span>
          </button>
          <span class="controls__balance" aria-hidden="true"></span>
        </nav>
      </section>

      <aside class="tools" id="tools" aria-label="Herramientas">
        <div class="tools__head">
          <h2 class="tools__title">Herramientas</h2>
          <button class="tools__close" id="tools-close" type="button" aria-label="Cerrar herramientas">
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor"
                 stroke-width="2" stroke-linecap="round" aria-hidden="true">
              <path d="M6 6l12 12M18 6 6 18" />
            </svg>
          </button>
        </div>

        <div class="tools__body">
          <div id="teach"></div>
          <div id="diagnostics"></div>
          <div id="storage"></div>

          <!--
            The reliability figure lives in the summary, not behind it: what made this section
            worth having was saying out loud that the model is fallible, so that sentence has to
            survive the fold.
          -->
          <details class="card">
            <summary class="card__summary">
              <h2 class="card__title">Qué reconoce</h2>
              <span class="card__note">286 signos LSE y el alfabeto — acierta 2 de cada 3 veces</span>
            </summary>

            <div class="card__content">
              <p class="card__body">
                <strong>Vocabulario LSE:</strong> 286 signos de ámbito sanitario, entrenados sobre
                SWL-LSE y LSE-Health. Signo a signo y sin prisa, acierta el signo exacto en torno a
                <strong>2 de cada 3 veces</strong>, y está entre sus tres primeras opciones en
                <strong>8 de cada 10</strong>.
              </p>
              <p class="card__body">
                <strong>Signando de corrido cae mucho:</strong> escribe la palabra correcta en torno a
                <strong>1 de cada 3</strong> signos. Un signo pegado al siguiente es más difícil que un
                signo suelto, y eso no está resuelto en ningún idioma todavía. Es un modelo real, no
                infalible: revisa el texto antes de darlo por bueno.
              </p>
              <p class="card__body">
                <strong>Alfabeto dactilológico:</strong> para deletrear cualquier palabra fuera de ese
                vocabulario. Intenta las 27 letras y acierta unas mejor que otras: desconfía de
                <strong>${WEAK_LETTERS.join(', ')}</strong>, que salen bien menos de una de cada tres
                veces porque apenas aparecen en el corpus con el que se entrenó.
              </p>
            </div>
          </details>

          <p class="footnote">
            v${__APP_VERSION__}
            <button class="theme" id="theme" type="button">
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor"
                   stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <g class="theme__icon" data-for="system">
                  <circle cx="12" cy="12" r="8" />
                  <path d="M12 4a8 8 0 0 1 0 16z" fill="currentColor" />
                </g>
                <g class="theme__icon" data-for="light">
                  <circle cx="12" cy="12" r="3.6" />
                  <path d="M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6l1.4 1.4M17 17l1.4 1.4M5.6 18.4 7 17M17 7l1.4-1.4" />
                </g>
                <g class="theme__icon" data-for="dark">
                  <path d="M19.5 14.2A7.8 7.8 0 0 1 9.8 4.5a7.8 7.8 0 1 0 9.7 9.7z" />
                </g>
              </svg>
            </button>
            · Vocabulario LSE sobre
            <a href="https://zenodo.org/records/13691887" rel="noreferrer">SWL-LSE</a> (CC-BY-4.0) y
            <a href="https://zenodo.org/records/10234465" rel="noreferrer">LSE-Health-UVigo</a>
            (CC-BY-NC-4.0) · el vídeo no sale de tu dispositivo:
            <a href="https://github.com/Endika/esku/blob/main/PRIVACY.md" rel="noreferrer">privacidad</a>
          </p>
        </div>
      </aside>
    </div>
  `;

  const video = must<HTMLVideoElement>(root, '#video');
  const overlayCanvas = must<HTMLCanvasElement>(root, '#overlay');
  const placeholder = must<HTMLElement>(root, '#placeholder');
  const hint = must<HTMLElement>(root, '#hint');
  const transcriptEl = must<HTMLElement>(root, '#transcript');
  const toggle = must<HTMLButtonElement>(root, '#toggle');
  const status = must<HTMLElement>(root, '#status');
  const edit = must<HTMLElement>(root, '#edit');
  const frame = must<HTMLElement>(root, '#frame');
  const toggleLabel = must<HTMLElement>(root, '#toggle-label');

  const themeButton = must<HTMLButtonElement>(root, '#theme');
  const THEME_NAMES: Record<ThemePreference, string> = {
    system: 'automático',
    light: 'claro',
    dark: 'oscuro',
  };
  let theme = readThemePreference();
  const showTheme = () => {
    themeButton.dataset.pref = theme;
    const label = `Tema ${THEME_NAMES[theme]}. Cambiar a ${THEME_NAMES[nextThemePreference(theme)]}`;
    themeButton.setAttribute('aria-label', label);
    themeButton.title = label;
  };
  themeButton.addEventListener('click', () => {
    theme = nextThemePreference(theme);
    applyThemePreference(theme);
    showTheme();
  });
  applyThemePreference(theme);
  followSystemTheme(() => theme);
  showTheme();

  // A sheet over the camera on a phone, a column beside it on a wide screen. Closed, it is
  // inert, so its controls drop out of the tab order and the screen reader alike.
  let running = false;
  const tools = must<HTMLElement>(root, '#tools');
  const toolsOpen = must<HTMLButtonElement>(root, '#tools-open');
  const docked = window.matchMedia('(min-width: 960px)');
  const setTools = (open: boolean) => {
    const shown = open || docked.matches;
    tools.classList.toggle('is-open', open);
    tools.inert = !shown;
    toolsOpen.setAttribute('aria-expanded', String(open));
  };
  toolsOpen.addEventListener('click', () => setTools(!tools.classList.contains('is-open')));
  must<HTMLButtonElement>(root, '#tools-close').addEventListener('click', () => {
    setTools(false);
    toolsOpen.focus();
  });
  root.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && tools.classList.contains('is-open')) {
      setTools(false);
      toolsOpen.focus();
    }
  });
  // Tapping the picture closes the sheet, as on the platform. Only the picture: the flip
  // button stays usable above an open sheet.
  frame.addEventListener('click', () => {
    if (tools.classList.contains('is-open')) setTools(false);
  });
  docked.addEventListener('change', () => setTools(false));
  setTools(false);

  // The brackets frame the picture, not the box around it: the same letterbox arithmetic the
  // overlay uses for `object-fit: contain`, so they never straddle the edge of the video.
  const placeGuides = () => {
    const { videoWidth, videoHeight } = video;
    if (!running || !videoWidth || !videoHeight) {
      frame.style.removeProperty('--guide-x');
      frame.style.removeProperty('--guide-y');
      return;
    }
    const scale = Math.min(frame.clientWidth / videoWidth, frame.clientHeight / videoHeight);
    frame.style.setProperty('--guide-x', `${(frame.clientWidth - videoWidth * scale) / 2}px`);
    frame.style.setProperty('--guide-y', `${(frame.clientHeight - videoHeight * scale) / 2}px`);
  };
  new ResizeObserver(placeGuides).observe(frame);
  video.addEventListener('loadedmetadata', placeGuides);
  video.addEventListener('resize', placeGuides);

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

  const parts = must<HTMLElement>(root, '#parts');
  const container = new Container(video);
  const { recognize } = container;
  const overlay = new LandmarkOverlay(overlayCanvas, video);
  const diagnostics = new DiagnosticsPanel(must<HTMLElement>(root, '#diagnostics'));

  /**
   * A tick when a part is being tracked, a cross when it is not, per part and never by colour
   * alone. The framing brackets sum it up: a hand found, then hands, face and torso in frame.
   */
  const showPresence = (presence: PartPresence | null) => {
    for (const part of PART_ORDER) {
      const chip = parts.querySelector<HTMLElement>(`[data-part="${part}"]`);
      const on = presence?.[part] === true;
      const off = presence !== null && presence[part] === false;
      chip?.classList.toggle('part--on', on);
      chip?.classList.toggle('part--off', off);
      const state = chip?.querySelector<HTMLElement>('.part__state');
      if (state) state.textContent = on ? ': visto' : off ? ': no visto' : '';
    }
    frame.dataset.track =
      presence === null
        ? 'idle'
        : presence.hands && presence.face && presence.torso
          ? 'framed'
          : presence.hands
            ? 'hand'
            : 'searching';
  };
  const render = (text: string, candidates: readonly { gloss: { text: string } }[]) => {
    const grew = text.length > (transcriptEl.textContent ?? '').length;
    transcriptEl.textContent = text;
    // The sign that just landed lifts out of the live guess and into the text.
    if (grew && !hint.hidden && !reduceMotion.matches) {
      hint.animate(
        [
          { transform: 'translate(-50%, 0)', opacity: 1 },
          { transform: 'translate(-50%, 28px)', opacity: 0 },
        ],
        { duration: 320, easing: 'cubic-bezier(0.16, 1, 0.3, 1)' },
      );
      transcriptEl.animate([{ opacity: 0.55 }, { opacity: 1 }], {
        duration: 420,
        easing: 'cubic-bezier(0.16, 1, 0.3, 1)',
      });
    }
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
      toggleLabel.textContent = 'Empezar a leer';
      placeholder.hidden = false;
      hint.hidden = true;
      video.classList.remove('is-live');
      overlayCanvas.classList.remove('is-live');
      overlay.clear();
      showPresence(null);
      placeGuides();
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
      placeGuides();
      root.classList.add('is-running');
      placeholder.hidden = true;
      video.classList.add('is-live');
      overlayCanvas.classList.add('is-live');
      toggleLabel.textContent = 'Parar';
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

  must<HTMLButtonElement>(root, '#transcript-undo').addEventListener('click', () => {
    recognize.undo();
    render(recognize.current.toText(), []);
  });

  must<HTMLButtonElement>(root, '#transcript-clear').addEventListener('click', () => {
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
