import {
  DuplicateSignTextError,
  NotEnoughExamplesError,
} from '@domain/recognition/entities/CustomSign';
import { SIGNATURE_LENGTH } from '@domain/recognition/services/windowSignature';
import { InMemoryCustomSignRepository } from '@infrastructure/persistence/InMemoryCustomSignRepository';
import { beforeEach, describe, expect, it } from 'vitest';
import { buildFrame, buildHand } from '@/test/handFixtures';
import { EmptySignTextError, TeachCustomSignUseCase } from '../TeachCustomSignUseCase';

function record(nudge = 0) {
  return Array.from({ length: 10 }, (_, i) =>
    buildFrame(i * 33, buildHand({ curls: [0.9, 0, 0.9, 0.9, 0.9], offset: { x: nudge, y: 0 } })),
  );
}

const EMPTY_RECORDING = [buildFrame(0, null), buildFrame(33, null)];

describe('TeachCustomSignUseCase', () => {
  let repository: InMemoryCustomSignRepository;
  let teach: TeachCustomSignUseCase;

  beforeEach(() => {
    repository = new InMemoryCustomSignRepository();
    teach = new TeachCustomSignUseCase(
      repository,
      () => 1_700_000_000_000,
      () => 'fixed-id',
    );
  });

  it('stores one prototype per usable recording', async () => {
    const sign = await teach.execute('ibuprofeno', [record(0), record(0.02), record(0.04)]);

    expect(sign.prototypes).toHaveLength(3);
    expect(sign.prototypes[0]).toHaveLength(SIGNATURE_LENGTH);
    expect(await repository.findAll()).toHaveLength(1);
  });

  it('records the injected id and timestamp', async () => {
    const sign = await teach.execute('ibuprofeno', [record(0), record(0.02), record(0.04)]);

    expect(sign.id).toBe('fixed-id');
    expect(sign.createdAtMs).toBe(1_700_000_000_000);
  });

  it('trims the text it will write', async () => {
    const sign = await teach.execute('  ibuprofeno  ', [record(0), record(0.02), record(0.04)]);
    expect(sign.text).toBe('ibuprofeno');
  });

  it('rejects a sign with no word to write', async () => {
    await expect(teach.execute('   ', [record(0), record(0.02), record(0.04)])).rejects.toThrow(
      EmptySignTextError,
    );
  });

  it('rejects fewer than three examples', async () => {
    await expect(teach.execute('ibuprofeno', [record(0), record(0.02)])).rejects.toThrow(
      NotEnoughExamplesError,
    );
  });

  it('does not count recordings that caught no hand', async () => {
    // Three attempts where two saw nothing is one example, not three. Accepting them would
    // store zero-filled prototypes that match almost anything.
    await expect(
      teach.execute('ibuprofeno', [record(0), EMPTY_RECORDING, EMPTY_RECORDING]),
    ).rejects.toThrow(NotEnoughExamplesError);
  });

  it('rejects a word that is already taught, whatever the casing', async () => {
    await teach.execute('ibuprofeno', [record(0), record(0.02), record(0.04)]);

    await expect(
      teach.execute('Ibuprofeno', [record(0), record(0.02), record(0.04)]),
    ).rejects.toThrow(DuplicateSignTextError);
  });

  it('leaves the repository untouched when it rejects', async () => {
    await expect(teach.execute('', [record(0), record(0.02), record(0.04)])).rejects.toThrow();
    expect(await repository.findAll()).toEqual([]);
  });
});
