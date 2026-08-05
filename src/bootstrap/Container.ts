import { ManageCustomSignsUseCase } from '@application/use-cases/ManageCustomSignsUseCase';
import { RecognizeSignsUseCase } from '@application/use-cases/RecognizeSignsUseCase';
import { TeachCustomSignUseCase } from '@application/use-cases/TeachCustomSignUseCase';
import type { ISignClassifier } from '@domain/recognition/services/ISignClassifier';
import { IndexedDBCustomSignRepository } from '@infrastructure/persistence/indexeddb/IndexedDBCustomSignRepository';
import { HandshapeAlphabetClassifier } from '@infrastructure/recognition/HandshapeAlphabetClassifier';
import { PrototypeSignClassifier } from '@infrastructure/recognition/PrototypeSignClassifier';
import { VocabularySignClassifier } from '@infrastructure/recognition/VocabularySignClassifier';
import { EngineCacheStorage } from '@infrastructure/storage/EngineCacheStorage';
import { MediaPipeLandmarkSource } from '@infrastructure/vision/MediaPipeLandmarkSource';

/**
 * The one place adapters meet use cases. No DI framework — constructor injection is enough
 * at this size, and it keeps the wiring readable in a single screen.
 */
export class Container {
  readonly source: MediaPipeLandmarkSource;
  readonly customSigns = new IndexedDBCustomSignRepository();
  readonly taught = new PrototypeSignClassifier(this.customSigns);
  readonly teach = new TeachCustomSignUseCase(this.customSigns);
  readonly manageCustomSigns = new ManageCustomSignsUseCase(this.customSigns);
  readonly engineStorage: EngineCacheStorage;
  readonly vocabulary: VocabularySignClassifier;
  readonly classifiers: readonly ISignClassifier[];
  readonly recognize: RecognizeSignsUseCase;

  constructor(video: HTMLVideoElement) {
    // BASE_URL, not a leading slash: the app is served from /esku/ on Pages and from / in dev.
    const base = import.meta.env.BASE_URL;
    this.engineStorage = new EngineCacheStorage(base);

    this.source = new MediaPipeLandmarkSource(video, {
      wasmPath: `${base}wasm`,
      handModelPath: `${base}models/hand_landmarker.task`,
      poseModelPath: `${base}models/pose_landmarker_lite.task`,
      faceModelPath: `${base}models/face_landmarker.task`,
      maxHands: 2,
    });

    this.vocabulary = new VocabularySignClassifier(
      `${base}models/lse-vocabulary.json`,
      `${base}models/lse-vocabulary.bin`,
    );

    this.classifiers = [new HandshapeAlphabetClassifier(), this.vocabulary, this.taught];
    this.recognize = new RecognizeSignsUseCase(this.source, this.classifiers);
  }
}
