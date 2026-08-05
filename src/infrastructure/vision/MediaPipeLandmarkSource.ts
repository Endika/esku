import {
  CameraUnavailableError,
  type ILandmarkSource,
  type LandmarkListener,
} from '@domain/landmarks/services/ILandmarkSource';
import {
  createHandLandmarks,
  type Handedness,
  type HandLandmarks,
} from '@domain/landmarks/value-objects/Landmark';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

export interface MediaPipeOptions {
  /** Where the vendored WASM lives, relative to the deployed base. */
  readonly wasmPath: string;
  readonly modelPath: string;
  readonly maxHands: number;
}

/**
 * Camera → hand landmarks, via MediaPipe Tasks running on WASM.
 *
 * Both the WASM runtime and the model are served same-origin from `public/`, never from a
 * CDN: this page holds camera permission, so letting a third party serve executable code
 * into it would be the single worst decision available. Offline support is a bonus of the
 * same choice.
 */
export class MediaPipeLandmarkSource implements ILandmarkSource {
  private landmarker: HandLandmarker | null = null;
  private stream: MediaStream | null = null;
  private frameHandle: number | null = null;
  private lastVideoTime = -1;
  private running = false;

  constructor(
    private readonly video: HTMLVideoElement,
    private readonly options: MediaPipeOptions,
  ) {}

  isRunning(): boolean {
    return this.running;
  }

  /**
   * Downloads and initialises the engine. Separate from `start` so the UI can show download
   * progress for the ~29 MB before asking for the camera — asking for permission and then
   * making the user wait reads as a hang.
   */
  async load(): Promise<void> {
    if (this.landmarker) return;
    const fileset = await FilesetResolver.forVisionTasks(this.options.wasmPath);
    this.landmarker = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: this.options.modelPath, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numHands: this.options.maxHands,
    });
  }

  async start(listener: LandmarkListener): Promise<void> {
    await this.load();

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
    } catch (cause) {
      throw new CameraUnavailableError(cause);
    }

    this.video.srcObject = this.stream;
    this.video.playsInline = true;
    this.video.muted = true;
    await this.video.play();

    this.running = true;
    this.pump(listener);
  }

  private pump(listener: LandmarkListener): void {
    const tick = () => {
      if (!this.running || !this.landmarker) return;

      // Detecting twice on the same decoded frame makes MediaPipe throw on the timestamp,
      // and wastes a GPU pass either way.
      if (this.video.currentTime !== this.lastVideoTime && this.video.readyState >= 2) {
        this.lastVideoTime = this.video.currentTime;
        const timestampMs = performance.now();
        const result = this.landmarker.detectForVideo(this.video, timestampMs);
        listener({ timestampMs, hands: toHands(result) });
      }

      this.frameHandle = requestAnimationFrame(tick);
    };
    this.frameHandle = requestAnimationFrame(tick);
  }

  stop(): void {
    this.running = false;
    if (this.frameHandle !== null) cancelAnimationFrame(this.frameHandle);
    this.frameHandle = null;
    this.lastVideoTime = -1;

    // Releasing the tracks is what turns the camera indicator off. Leaving them live would
    // keep recording in the background, which this app must never appear to do.
    this.stream?.getTracks().forEach((track) => {
      track.stop();
    });
    this.stream = null;
    this.video.srcObject = null;
  }
}

interface MediaPipeResult {
  readonly landmarks?: { x: number; y: number; z: number }[][];
  readonly handedness?: { categoryName?: string }[][];
}

function toHands(result: MediaPipeResult): HandLandmarks[] {
  const hands: HandLandmarks[] = [];
  result.landmarks?.forEach((points, i) => {
    const label = result.handedness?.[i]?.[0]?.categoryName;
    // MediaPipe labels the *mirrored* selfie view, so its "Left" is the user's right hand.
    const handedness: Handedness = label === 'Left' ? 'right' : 'left';
    hands.push(createHandLandmarks(handedness, points));
  });
  return hands;
}
