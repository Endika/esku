declare const __APP_VERSION__: string;

/**
 * The app shell: masthead, camera stage, transcript and actions.
 *
 * Deliberately dumb for now — it renders the surface and owns no recognition logic. The
 * camera adapter and the classifiers plug in behind their ports, so wiring them up changes
 * this file's event handlers and nothing about the layout.
 */
export function renderApp(root: HTMLElement): void {
  root.innerHTML = `
    <header class="masthead">
      <img class="masthead__mark" src="favicon.svg" alt="" width="44" height="44" />
      <div>
        <h1 class="masthead__title">Esku</h1>
        <p class="masthead__tagline">Lengua de signos a texto, sin conexión</p>
      </div>
    </header>

    <div class="stage" id="stage">
      <p class="stage__placeholder">
        La cámara se activa al empezar. El vídeo no se graba ni se envía a ningún sitio.
      </p>
    </div>

    <div class="transcript" id="transcript" aria-live="polite" aria-atomic="false"></div>

    <div class="actions">
      <button class="button" id="start" type="button" disabled>Empezar a leer</button>
      <button class="button button--quiet" id="teach" type="button" disabled>
        Enseñar un signo
      </button>
    </div>

    <section class="card" style="margin-top: 24px">
      <h2 class="card__title">Qué reconoce <span class="badge">en construcción</span></h2>
      <p class="card__body">
        Alfabeto dactilológico para deletrear, vocabulario de signos completos en LSE, y los
        signos que le enseñes tú. Todo el reconocimiento ocurre en el dispositivo.
      </p>
    </section>

    <p class="footnote">
      v${__APP_VERSION__} · Vocabulario LSE entrenado sobre
      <a href="https://zenodo.org/records/13691887" rel="noreferrer">SWL-LSE</a> (CC-BY-4.0)
    </p>
  `;
}
