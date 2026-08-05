import { TeachCustomSignUseCase } from '@application/use-cases/TeachCustomSignUseCase';
import type { LandmarkFrame } from '@domain/landmarks/value-objects/LandmarkFrame';
import { InMemoryCustomSignRepository } from '@infrastructure/persistence/InMemoryCustomSignRepository';
import { beforeEach, describe, expect, it } from 'vitest';
import { buildFrame, buildHand } from '@/test/handFixtures';
import { PrototypeSignClassifier } from '../PrototypeSignClassifier';

type Curls = [number, number, number, number, number];

const FIST: Curls = [0.9, 0.9, 0.9, 0.9, 0.9];
const OPEN: Curls = [0, 0, 0, 0, 0];
const Y: Curls = [0, 0.9, 0.9, 0.9, 0];

/** One recording, with a nudge so repeated examples are not byte-identical. */
function record(curls: Curls, nudge = 0): readonly LandmarkFrame[] {
  return Array.from({ length: 12 }, (_, i) =>
    buildFrame(
      i * 33,
      buildHand({
        curls: curls.map((c) => Math.max(0, Math.min(1, c + nudge))) as Curls,
        offset: { x: nudge, y: nudge / 2 },
      }),
    ),
  );
}

function threeTakesOf(curls: Curls) {
  return [record(curls, 0), record(curls, 0.03), record(curls, -0.03)];
}

describe('PrototypeSignClassifier', () => {
  let repository: InMemoryCustomSignRepository;
  let teach: TeachCustomSignUseCase;
  let classifier: PrototypeSignClassifier;
  let clock = 0;

  beforeEach(() => {
    clock = 0;
    repository = new InMemoryCustomSignRepository();
    teach = new TeachCustomSignUseCase(
      repository,
      () => {
        clock += 1000;
        return clock;
      },
      () => `sign-${clock}`,
    );
    classifier = new PrototypeSignClassifier(repository);
  });

  it('is not ready before anything has been taught', async () => {
    await classifier.load();
    expect(classifier.isReady()).toBe(false);
  });

  it('stays silent when nothing has been taught', async () => {
    await classifier.load();
    expect(await classifier.classify(record(FIST))).toEqual([]);
  });

  it('recognises a sign it was taught', async () => {
    await teach.execute('ibuprofeno', threeTakesOf(FIST));
    await classifier.load();

    const [top] = await classifier.classify(record(FIST, 0.02));
    expect(top?.gloss.text).toBe('ibuprofeno');
    expect(top?.source).toBe('taught');
  });

  it('picks the right sign out of several', async () => {
    await teach.execute('ibuprofeno', threeTakesOf(FIST));
    await teach.execute('paracetamol', threeTakesOf(OPEN));
    await teach.execute('receta', threeTakesOf(Y));
    await classifier.load();

    const [top] = await classifier.classify(record(OPEN, 0.02));
    expect(top?.gloss.text).toBe('paracetamol');
  });

  it('declines a sign it was never taught', async () => {
    await teach.execute('ibuprofeno', threeTakesOf(OPEN));
    await classifier.load();

    expect(await classifier.classify(record(Y))).toEqual([]);
  });

  it('sees a newly taught sign only after a refresh', async () => {
    await classifier.load();
    await teach.execute('ibuprofeno', threeTakesOf(FIST));

    expect(await classifier.classify(record(FIST))).toEqual([]);
    await classifier.refresh();
    expect((await classifier.classify(record(FIST)))[0]?.gloss.text).toBe('ibuprofeno');
  });

  it('stops recognising a deleted sign after a refresh', async () => {
    const sign = await teach.execute('ibuprofeno', threeTakesOf(FIST));
    await classifier.load();
    await repository.delete(sign.id);
    await classifier.refresh();

    expect(await classifier.classify(record(FIST))).toEqual([]);
  });

  it('reads windows, not frames, so the segmenter drives it', () => {
    expect(classifier.granularity).toBe('window');
  });

  it('ignores an empty window', async () => {
    await teach.execute('ibuprofeno', threeTakesOf(FIST));
    await classifier.load();
    expect(await classifier.classify([])).toEqual([]);
  });
});
