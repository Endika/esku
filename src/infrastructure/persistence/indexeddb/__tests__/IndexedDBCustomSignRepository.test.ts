import type { CustomSign } from '@domain/recognition/entities/CustomSign';
import { IDBFactory } from 'fake-indexeddb';
import { beforeEach, describe, expect, it } from 'vitest';
import { IndexedDBCustomSignRepository } from '../IndexedDBCustomSignRepository';

function sign(id: string, text: string, createdAtMs: number): CustomSign {
  return {
    id,
    text,
    prototypes: [Float32Array.from([1, 2, 3]), Float32Array.from([4, 5, 6])],
    createdAtMs,
  };
}

describe('IndexedDBCustomSignRepository', () => {
  let repository: IndexedDBCustomSignRepository;

  beforeEach(() => {
    // A fresh factory per test, so one test's signs cannot leak into the next.
    globalThis.indexedDB = new IDBFactory();
    repository = new IndexedDBCustomSignRepository();
  });

  it('reads back a sign it saved', async () => {
    await repository.save(sign('a', 'ibuprofeno', 1));
    const [stored] = await repository.findAll();
    expect(stored?.text).toBe('ibuprofeno');
  });

  it('keeps prototypes as binary typed arrays rather than plain arrays', async () => {
    // The whole reason for IndexedDB over localStorage: no JSON round-trip, no type loss.
    //
    // Checked with `ArrayBuffer.isView` rather than `instanceof Float32Array`: structured
    // clone in fake-indexeddb rebuilds the array in another realm, so it is a genuine
    // Float32Array that fails an identity check against this realm's constructor. That is
    // a property of the test environment, not of the stored data.
    await repository.save(sign('a', 'ibuprofeno', 1));
    const [stored] = await repository.findAll();
    const prototype = stored?.prototypes[0];

    expect(ArrayBuffer.isView(prototype)).toBe(true);
    expect(prototype?.constructor.name).toBe('Float32Array');
    expect(Array.from(prototype ?? [])).toEqual([1, 2, 3]);
    expect(stored?.prototypes).toHaveLength(2);
  });

  it('returns signs oldest first', async () => {
    await repository.save(sign('b', 'segundo', 200));
    await repository.save(sign('a', 'primero', 100));

    expect((await repository.findAll()).map((s) => s.text)).toEqual(['primero', 'segundo']);
  });

  it('overwrites a sign saved under the same id', async () => {
    await repository.save(sign('a', 'ibuprofeno', 1));
    await repository.save(sign('a', 'paracetamol', 1));

    const all = await repository.findAll();
    expect(all).toHaveLength(1);
    expect(all[0]?.text).toBe('paracetamol');
  });

  it('deletes a sign', async () => {
    await repository.save(sign('a', 'ibuprofeno', 1));
    await repository.save(sign('b', 'paracetamol', 2));
    await repository.delete('a');

    expect((await repository.findAll()).map((s) => s.text)).toEqual(['paracetamol']);
  });

  it('treats deleting something absent as done, not as an error', async () => {
    await expect(repository.delete('never-existed')).resolves.toBeUndefined();
  });

  it('starts empty', async () => {
    expect(await repository.findAll()).toEqual([]);
  });

  it('survives concurrent saves without racing the database open', async () => {
    // Every call opens the same cached connection; if that cache were missing, parallel
    // opens would race the upgrade and one would fail.
    await Promise.all([
      repository.save(sign('a', 'uno', 1)),
      repository.save(sign('b', 'dos', 2)),
      repository.save(sign('c', 'tres', 3)),
    ]);

    expect(await repository.findAll()).toHaveLength(3);
  });
});
