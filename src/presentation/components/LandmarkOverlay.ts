import { HAND_CONNECTIONS } from '@domain/landmarks/value-objects/Landmark';
import type { LandmarkFrame } from '@domain/landmarks/value-objects/LandmarkFrame';

/** How the overlay reads back to the user. Colour is the whole feedback channel here. */
export type OverlayState = 'searching' | 'tracking' | 'recognised';

const COLOURS: Record<OverlayState, { line: string; joint: string }> = {
  // Nothing found: nothing drawn, so these are only a fallback.
  searching: { line: 'rgba(165, 160, 192, 0.7)', joint: 'rgba(165, 160, 192, 0.9)' },
  // Hand seen but no confident letter — the honest "I see you, I don't know that shape".
  tracking: { line: 'rgba(245, 243, 255, 0.55)', joint: 'rgba(245, 243, 255, 0.85)' },
  // A letter is being read right now.
  recognised: { line: 'rgba(167, 139, 250, 0.95)', joint: '#c4b5fd' },
};

/**
 * Draws the tracked hand skeleton over the camera feed.
 *
 * This is the app's only honest answer to "is it seeing me?". Without it a failure to
 * recognise is indistinguishable from a failure to detect, and the user has no idea whether
 * to move closer, improve the light or change the handshape.
 *
 * Geometry note: the canvas is CSS-mirrored to match the video, and landmarks are mapped
 * through the same `object-fit: contain` letterboxing the video uses, so the skeleton lands
 * on the hand rather than near it.
 */
export class LandmarkOverlay {
  private readonly context: CanvasRenderingContext2D;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly video: HTMLVideoElement,
  ) {
    const context = canvas.getContext('2d');
    if (!context) throw new Error('Canvas 2D context unavailable');
    this.context = context;
  }

  clear(): void {
    this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  draw(frame: LandmarkFrame, state: OverlayState): void {
    this.resizeToDisplay();
    this.clear();
    if (frame.hands.length === 0) return;

    const { line, joint } = COLOURS[state];
    const project = this.projector();
    // Scale strokes with the canvas so the skeleton looks the same on a phone and a laptop.
    const unit = Math.max(this.canvas.width, this.canvas.height) / 320;

    this.context.lineCap = 'round';
    this.context.lineJoin = 'round';

    for (const hand of frame.hands) {
      this.context.strokeStyle = line;
      this.context.lineWidth = 3 * unit;
      for (const [from, to] of HAND_CONNECTIONS) {
        const a = hand.points[from];
        const b = hand.points[to];
        if (!a || !b) continue;
        const start = project(a.x, a.y);
        const end = project(b.x, b.y);
        this.context.beginPath();
        this.context.moveTo(start.x, start.y);
        this.context.lineTo(end.x, end.y);
        this.context.stroke();
      }

      this.context.fillStyle = joint;
      for (const point of hand.points) {
        const { x, y } = project(point.x, point.y);
        this.context.beginPath();
        this.context.arc(x, y, 2.6 * unit, 0, Math.PI * 2);
        this.context.fill();
      }
    }
  }

  /** Backing store in device pixels, so the skeleton is not blurry on a phone. */
  private resizeToDisplay(): void {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.round(this.canvas.clientWidth * ratio);
    const height = Math.round(this.canvas.clientHeight * ratio);
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
  }

  /**
   * Maps normalised landmark coordinates to canvas pixels through the same letterboxing
   * `object-fit: contain` applies, so the skeleton tracks the hand exactly.
   */
  private projector(): (x: number, y: number) => { x: number; y: number } {
    const videoWidth = this.video.videoWidth || this.canvas.width;
    const videoHeight = this.video.videoHeight || this.canvas.height;
    const scale = Math.min(this.canvas.width / videoWidth, this.canvas.height / videoHeight);
    const drawnWidth = videoWidth * scale;
    const drawnHeight = videoHeight * scale;
    const offsetX = (this.canvas.width - drawnWidth) / 2;
    const offsetY = (this.canvas.height - drawnHeight) / 2;

    return (x, y) => ({ x: offsetX + x * drawnWidth, y: offsetY + y * drawnHeight });
  }
}
