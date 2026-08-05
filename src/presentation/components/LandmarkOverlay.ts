import {
  ARM_CONNECTIONS,
  NECK_CONNECTIONS,
  TORSO_CONNECTIONS,
} from '@domain/landmarks/value-objects/BodyLandmarks';
import { HAND_CONNECTIONS, type Landmark } from '@domain/landmarks/value-objects/Landmark';
import type { LandmarkFrame } from '@domain/landmarks/value-objects/LandmarkFrame';

/** How the overlay reads back to the user. Colour is the whole feedback channel here. */
export type OverlayState = 'searching' | 'tracking' | 'recognised';

/** The parts drawn, each in its own colour so it is obvious what is and is not being seen. */
export type BodyPart = 'hands' | 'face' | 'neck' | 'torso' | 'arms';

export const PART_COLOURS: Record<BodyPart, string> = {
  hands: '#a78bfa',
  face: '#22d3ee',
  neck: '#f472b6',
  torso: '#fbbf24',
  arms: '#4ade80',
};

export const PART_LABELS: Record<BodyPart, string> = {
  hands: 'Manos',
  face: 'Cara',
  neck: 'Cuello',
  torso: 'Torso',
  arms: 'Brazos',
};

export const PART_ORDER: readonly BodyPart[] = ['hands', 'face', 'neck', 'torso', 'arms'];

/** Which parts were visible in the last drawn frame, for the status chips. */
export type PartPresence = Record<BodyPart, boolean>;

/**
 * Draws everything the app is tracking, over the camera feed.
 *
 * This is the app's only honest answer to "is it seeing me?". Without it, a failure to
 * recognise is indistinguishable from a failure to detect, and the user has no idea whether
 * to move closer, improve the light, or change the handshape.
 *
 * Each body part gets its own colour rather than one uniform skeleton, because the parts
 * fail independently: hands track at arm's length while the face is out of frame, or the
 * torso is cropped while everything else is fine. One colour would hide which.
 *
 * Geometry note: the canvas is CSS-mirrored to match the video, and landmarks are mapped
 * through the same `object-fit: contain` letterboxing, so a skeleton lands on the body
 * rather than near it.
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

  /** Returns which parts it actually found, so the UI can flag the missing ones. */
  draw(frame: LandmarkFrame, state: OverlayState): PartPresence {
    this.resizeToDisplay();
    this.clear();

    const project = this.projector();
    const unit = Math.max(this.canvas.width, this.canvas.height) / 320;
    // A recognised sign brightens everything; while merely tracking, the skeleton stays
    // quiet so it does not compete with the video for attention.
    const alpha = state === 'recognised' ? 1 : 0.7;

    this.context.lineCap = 'round';
    this.context.lineJoin = 'round';
    this.context.globalAlpha = alpha;

    const pose = frame.pose?.points ?? [];
    const face = frame.face?.points ?? [];

    if (pose.length > 0) {
      this.drawEdges(pose, TORSO_CONNECTIONS, PART_COLOURS.torso, 3.5 * unit, project);
      this.drawEdges(pose, ARM_CONNECTIONS, PART_COLOURS.arms, 3.5 * unit, project);
      this.drawEdges(pose, NECK_CONNECTIONS, PART_COLOURS.neck, 2.5 * unit, project);
    }

    if (face.length > 0) {
      // Points only: the mesh's own edges are far too dense to read at video size, and a
      // scatter already shows the face is tracked.
      this.drawPoints(face, PART_COLOURS.face, 1.1 * unit, project);
    }

    for (const hand of frame.hands) {
      this.drawEdges(hand.points, HAND_CONNECTIONS, PART_COLOURS.hands, 3 * unit, project);
      this.drawPoints(hand.points, PART_COLOURS.hands, 2.4 * unit, project);
    }

    this.context.globalAlpha = 1;

    return {
      hands: frame.hands.length > 0,
      face: face.length > 0,
      neck: pose.length > 0,
      torso: pose.length > 0,
      arms: pose.length > 0,
    };
  }

  private drawEdges(
    points: readonly Landmark[],
    edges: readonly (readonly [number, number])[],
    colour: string,
    width: number,
    project: (x: number, y: number) => { x: number; y: number },
  ): void {
    this.context.strokeStyle = colour;
    this.context.lineWidth = width;
    for (const [from, to] of edges) {
      const a = points[from];
      const b = points[to];
      if (!a || !b) continue;
      const start = project(a.x, a.y);
      const end = project(b.x, b.y);
      this.context.beginPath();
      this.context.moveTo(start.x, start.y);
      this.context.lineTo(end.x, end.y);
      this.context.stroke();
    }
  }

  private drawPoints(
    points: readonly Landmark[],
    colour: string,
    radius: number,
    project: (x: number, y: number) => { x: number; y: number },
  ): void {
    this.context.fillStyle = colour;
    for (const point of points) {
      const { x, y } = project(point.x, point.y);
      this.context.beginPath();
      this.context.arc(x, y, radius, 0, Math.PI * 2);
      this.context.fill();
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
   * `object-fit: contain` applies, so the skeleton tracks the body exactly.
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
