import {
  CameraUnavailableError,
  type ILandmarkSource,
  type LandmarkListener,
} from '@domain/landmarks/services/ILandmarkSource';
import { PosePoint } from '@domain/landmarks/value-objects/BodyLandmarks';
import { type FrameCost, FrameCostMeter } from '@domain/landmarks/value-objects/FrameCost';
import {
  createHandLandmarks,
  type Handedness,
  type HandLandmarks,
} from '@domain/landmarks/value-objects/Landmark';
import type { LandmarkFrame } from '@domain/landmarks/value-objects/LandmarkFrame';
import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
} from '@mediapipe/tasks-vision';

export type CameraFacing = 'user' | 'environment';

export interface MediaPipeOptions {
  /** Where the vendored WASM lives, relative to the deployed base. */
  readonly wasmPath: string;
  readonly handModelPath: string;
  readonly poseModelPath: string;
  readonly faceModelPath: string;
  readonly maxHands: number;
  /**
   * How often to run the face detector, in milliseconds.
   *
   * Non-manual markers are slow: a raised brow marking a question, a negating head shake, a
   * mouth gesture — all last half a second or more, so sampling them once per frame was paying
   * a whole model per frame for a signal that does not move that fast. Measured against the
   * released weights, holding the last reading for a full second costs **0.001 top-1** while
   * removing the block entirely costs 0.008. Frame rate is what decides whether this app writes
   * anything at all, so the third model runs on its own clock.
   */
  readonly faceIntervalMs: number;
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
  private pose: PoseLandmarker | null = null;
  private face: FaceLandmarker | null = null;
  private stream: MediaStream | null = null;
  private frameHandle: number | null = null;
  private lastVideoTime = -1;
  private running = false;
  private facing: CameraFacing = 'user';
  private lastFaceMs = Number.NEGATIVE_INFINITY;
  /**
   * Sticky, because a signer turning sideways for a moment must not flip every hand. Starts
   * `false`: three independent measurements say MediaPipe's raw label is already anatomical, so
   * "no mirror" is the measured default rather than a hopeful one.
   */
  private mirrored = false;
  private heldFace: LandmarkFrame['face'];
  private listener: LandmarkListener | null = null;
  private readonly cost = new FrameCostMeter();

  constructor(
    private readonly video: HTMLVideoElement,
    private readonly options: MediaPipeOptions,
  ) {}

  isRunning(): boolean {
    return this.running;
  }

  frameCost(): FrameCost | null {
    return this.cost.read();
  }

  get camera(): CameraFacing {
    return this.facing;
  }

  /**
   * Front camera for signing to yourself, rear for reading someone else.
   *
   * Restarts the stream rather than reconfiguring it: `getUserMedia` cannot change
   * `facingMode` on a live track, and phones expose the two cameras as separate devices.
   */
  async useCamera(facing: CameraFacing): Promise<void> {
    if (facing === this.facing) return;
    this.facing = facing;

    if (!this.running) return;
    const listener = this.listener;
    this.stop();
    if (listener) await this.start(listener);
  }

  /**
   * Downloads and initialises the engine. Separate from `start` so the UI can show download
   * progress for the ~29 MB before asking for the camera — asking for permission and then
   * making the user wait reads as a hang.
   */
  async load(): Promise<void> {
    if (this.landmarker) return;
    const fileset = await FilesetResolver.forVisionTasks(this.options.wasmPath);

    // Hands first and awaited alone: they are the only model recognition cannot work
    // without, so a failure here is fatal while the other two are enhancements.
    this.landmarker = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: this.options.handModelPath, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numHands: this.options.maxHands,
    });

    const [pose, face] = await Promise.all([
      PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: this.options.poseModelPath, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numPoses: 1,
      }).catch(() => null),
      FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: this.options.faceModelPath, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numFaces: 1,
      }).catch(() => null),
    ]);
    this.pose = pose;
    this.face = face;
  }

  async start(listener: LandmarkListener): Promise<void> {
    await this.load();
    this.listener = listener;

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: this.facing, width: { ideal: 1280 }, height: { ideal: 720 } },
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
        const afterHands = performance.now();
        const pose = this.pose?.detectForVideo(this.video, timestampMs);
        const afterPose = performance.now();
        const dueFace = timestampMs - this.lastFaceMs >= this.options.faceIntervalMs;
        let faceMs: number | null = null;
        if (dueFace) {
          const face = this.face?.detectForVideo(this.video, timestampMs);
          faceMs = performance.now() - afterPose;
          this.lastFaceMs = timestampMs;
          this.heldFace = face?.faceLandmarks?.[0] ? { points: face.faceLandmarks[0] } : undefined;
        }

        // VIDEO mode returns the result, so the call has waited for its own GPU work
        this.cost.record(afterHands - timestampMs, afterPose - afterHands, faceMs);

        const seen = mirroringFrom(pose?.landmarks?.[0]);
        if (seen !== 'unknown') this.mirrored = seen === 'mirrored';

        listener({
          timestampMs,
          hands: toHands(result, this.mirrored),
          pose: pose?.landmarks?.[0] ? { points: pose.landmarks[0] } : undefined,
          face: this.heldFace,
        });
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
    // A held face from the previous session would be the wrong person's, or the wrong camera's
    this.lastFaceMs = Number.NEGATIVE_INFINITY;
    this.heldFace = undefined;
    this.mirrored = false;
    // Front and rear cameras do not cost the same, and useCamera() switches through here
    this.cost.reset();

    // Releasing the tracks is what turns the camera indicator off. Leaving them live would
    // keep recording in the background, which this app must never appear to do.
    this.stream?.getTracks().forEach((track) => {
      track.stop();
    });
    this.stream = null;
    this.video.srcObject = null;
  }

  /** Which extra trackers came up. The UI reports this so a missing part is explainable. */
  get available(): { pose: boolean; face: boolean } {
    return { pose: this.pose !== null, face: this.face !== null };
  }
}

interface MediaPipeResult {
  readonly landmarks?: { x: number; y: number; z: number }[][];
  readonly handedness?: { categoryName?: string }[][];
}

/** Below this normalised shoulder separation the signer is turned too far to tell. */
const MIN_SHOULDER_GAP = 0.05;

/**
 * Is the frame a mirror image, judged from the body rather than assumed from the camera?
 *
 * Pose labels shoulders anatomically, so the geometry answers on its own: someone facing the
 * lens has their anatomical left shoulder on the viewer's right, which is the larger x. If the
 * order is reversed, the frame is mirrored. No device list, no guessing.
 *
 * This replaces "the front camera is mirrored", which was written down, unit-tested, and wrong.
 * `getUserMedia` hands over the sensor's own frames; the selfie mirror is a CSS convention on
 * the preview, so MediaPipe never saw a mirrored image and its label was anatomical all along.
 * Measured three ways: against Pose's own wrist on LSE-Health (98.6% agreement), against image
 * position on CALSE100, and finally on a real phone, where a right hand was landing in the left
 * slot — and `normalizeHand` mirrors left hands into the right hand's space, so every
 * right-handed signer was handing the model a reflected handshape.
 */
export function mirroringFrom(
  pose: readonly { x: number }[] | undefined,
): 'mirrored' | 'direct' | 'unknown' {
  const left = pose?.[PosePoint.leftShoulder];
  const right = pose?.[PosePoint.rightShoulder];
  if (!left || !right) return 'unknown';
  const gap = left.x - right.x;
  if (Math.abs(gap) < MIN_SHOULDER_GAP) return 'unknown';
  return gap > 0 ? 'direct' : 'mirrored';
}

/**
 * Which of the signer's hands MediaPipe just labelled.
 *
 * Getting this wrong does not swap hands, it corrupts them: `normalizeHand` reflects left hands
 * into the right hand's space, so a hand in the wrong slot reaches the model inside out.
 *
 * Exported so the reasoning can be tested without a camera.
 */
export function handednessFor(label: string | undefined, mirrored: boolean): Handedness {
  const isRight = mirrored ? label === 'Left' : label === 'Right';
  return isRight ? 'right' : 'left';
}

function toHands(result: MediaPipeResult, mirrored: boolean): HandLandmarks[] {
  const hands: HandLandmarks[] = [];
  result.landmarks?.forEach((points, i) => {
    const label = result.handedness?.[i]?.[0]?.categoryName;
    hands.push(createHandLandmarks(handednessFor(label, mirrored), points));
  });
  return hands;
}
