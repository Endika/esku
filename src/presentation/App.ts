import { Container } from '@bootstrap/Container';
import { CameraUnavailableError } from '@domain/landmarks/services/ILandmarkSource';
import { UNSUPPORTED_LETTERS } from '@infrastructure/recognition/packs/lseAlphabet';

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
      <p class="stage__placeholder" id="placeholder">
        La cámara se activa al empezar.<br />El vídeo no se graba ni sale del dispositivo.
      </p>
      <p class="stage__hint" id="hint" hidden></p>
    </div>

    <div class="transcript" id="transcript" aria-live="polite"></div>

    <div class="actions">
      <button class="button" id="toggle" type="button">Empezar a leer</button>
      <button class="button button--quiet" id="undo" type="button">Borrar último</button>
      <button class="button button--quiet" id="clear" type="button">Limpiar</button>
    </div>

    <p class="status" id="status" role="status"></p>

    <section class="card">
      <h2 class="card__title">Deletreo, por ahora</h2>
      <p class="card__body">
        Reconoce las letras del dactilológico que se distinguen por la forma de la mano.
        Todavía no distingue <strong>${UNSUPPORTED_LETTERS.join(', ')}</strong>: unas se
        trazan con movimiento y otras dependen de la orientación de la palma. El vocabulario
        de signos completos llega con el modelo entrenado.
      </p>
    </section>

    <p class="footnote">
      v${__APP_VERSION__} · Vocabulario LSE sobre
      <a href="https://zenodo.org/records/13691887" rel="noreferrer">SWL-LSE</a> (CC-BY-4.0)
    </p>
  `;

  const video = must<HTMLVideoElement>(root, '#video');
  const placeholder = must<HTMLElement>(root, '#placeholder');
  const hint = must<HTMLElement>(root, '#hint');
  const transcriptEl = must<HTMLElement>(root, '#transcript');
  const toggle = must<HTMLButtonElement>(root, '#toggle');
  const status = must<HTMLElement>(root, '#status');

  const container = new Container(video);
  const { recognize } = container;
  let running = false;

  const render = (text: string, candidates: readonly { gloss: { text: string } }[]) => {
    transcriptEl.textContent = text;
    const top = candidates[0]?.gloss.text;
    hint.hidden = !top;
    if (top) hint.textContent = top.toUpperCase();
  };

  toggle.addEventListener('click', async () => {
    if (running) {
      recognize.stop();
      running = false;
      toggle.textContent = 'Empezar a leer';
      placeholder.hidden = false;
      hint.hidden = true;
      video.classList.remove('is-live');
      status.textContent = 'Cámara apagada.';
      return;
    }

    toggle.disabled = true;
    // The engine is ~29 MB on first run and cached after; say so rather than look frozen.
    status.textContent = 'Preparando el motor de reconocimiento…';
    try {
      await recognize.start(({ transcript, candidates }) => {
        render(transcript.toText(), candidates);
      });
      running = true;
      placeholder.hidden = true;
      video.classList.add('is-live');
      toggle.textContent = 'Parar';
      status.textContent = 'Leyendo. Mantén la mano dentro del encuadre.';
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
}

function must<T extends Element>(root: ParentNode, selector: string): T {
  const found = root.querySelector<T>(selector);
  if (!found) throw new Error(`Missing element: ${selector}`);
  return found;
}
