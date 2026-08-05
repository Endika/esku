import { formatBytes } from '@infrastructure/storage/EngineCacheStorage';

export interface StoragePanelPorts {
  isSupported(): boolean;
  report(): Promise<{ cachedBytes: number; entries: number }>;
  clear(): Promise<boolean>;
  /** Fetches the engine now, so the first real use is not the first download. */
  preload(): Promise<void>;
}

/**
 * Lets the user see and reclaim the space the recognition engine takes.
 *
 * Downloading is offered up front rather than only happening on first use: knowing it is
 * ~29 MB before it starts is the difference between a considered choice and a surprise on
 * mobile data.
 */
export class StoragePanel {
  private busy = false;

  constructor(
    private readonly root: HTMLElement,
    private readonly ports: StoragePanelPorts,
  ) {
    this.render();
    void this.refresh();
  }

  private render(): void {
    this.root.innerHTML = `
      <section class="card">
        <h2 class="card__title">Espacio en el dispositivo</h2>
        <p class="card__body">
          El motor de reconocimiento ocupa unos 41 MB y se guarda la primera vez que lo usas,
          para que después funcione sin conexión. Puedes descargarlo ahora o liberarlo cuando
          quieras: se volverá a bajar solo la próxima vez que enciendas la cámara.
        </p>

        <p class="storage" id="storage-figure">—</p>

        <div class="actions">
          <button class="button button--quiet" id="preload" type="button">Descargar ahora</button>
          <button class="button button--quiet" id="clear" type="button">Liberar espacio</button>
        </div>

        <p class="status" id="storage-status" role="status"></p>
        <p class="card__body card__body--tight">
          Los signos que le hayas enseñado no se borran con esto: son tuyos y se guardan aparte.
        </p>
      </section>
    `;

    this.button('preload').addEventListener('click', () => void this.preload());
    this.button('clear').addEventListener('click', () => void this.clear());
  }

  private button(id: string): HTMLButtonElement {
    return this.root.querySelector<HTMLButtonElement>(`#${id}`)!;
  }

  private say(message: string): void {
    this.root.querySelector<HTMLElement>('#storage-status')!.textContent = message;
  }

  private setBusy(busy: boolean): void {
    this.busy = busy;
    this.button('preload').disabled = busy;
    this.button('clear').disabled = busy;
  }

  private async preload(): Promise<void> {
    if (this.busy) return;
    this.setBusy(true);
    this.say('Descargando el motor…');
    try {
      await this.ports.preload();
      this.say('Motor descargado. Ya funciona sin conexión.');
    } catch {
      this.say('No se pudo descargar. Comprueba la conexión.');
    } finally {
      this.setBusy(false);
      await this.refresh();
    }
  }

  private async clear(): Promise<void> {
    if (this.busy) return;
    this.setBusy(true);
    try {
      const removed = await this.ports.clear();
      this.say(
        removed
          ? 'Espacio liberado. Se volverá a descargar la próxima vez.'
          : 'No había nada guardado.',
      );
    } finally {
      this.setBusy(false);
      await this.refresh();
    }
  }

  private async refresh(): Promise<void> {
    const figure = this.root.querySelector<HTMLElement>('#storage-figure')!;

    if (!this.ports.isSupported()) {
      figure.textContent = 'Este navegador no permite consultar el almacenamiento.';
      this.setBusy(true);
      return;
    }

    const { cachedBytes, entries } = await this.ports.report();
    figure.textContent =
      entries === 0 ? 'Nada guardado todavía' : `${formatBytes(cachedBytes)} guardados`;
  }
}
