import type { CustomSign } from '@domain/recognition/entities/CustomSign';
import type { ICustomSignRepository } from '@domain/recognition/repositories/ICustomSignRepository';

const DATABASE = 'esku';
const STORE = 'custom-signs';
const VERSION = 1;

/**
 * Taught signs, persisted on the device.
 *
 * IndexedDB rather than localStorage because prototypes are `Float32Array`s: IndexedDB
 * stores them via structured clone with no conversion, while localStorage would force a
 * JSON round-trip through arrays of numbers — several times the size, and lossy on the
 * type.
 *
 * These recordings are derived from the user's own body and may encode health vocabulary,
 * so they stay here: same-origin, on-device, never synced.
 */
export class IndexedDBCustomSignRepository implements ICustomSignRepository {
  private database: Promise<IDBDatabase> | null = null;

  async save(sign: CustomSign): Promise<void> {
    const database = await this.open();
    await run(database, 'readwrite', (store) => store.put(sign));
  }

  async findAll(): Promise<CustomSign[]> {
    const database = await this.open();
    const signs = await run<CustomSign[]>(database, 'readonly', (store) => store.getAll());
    return signs.sort((a, b) => a.createdAtMs - b.createdAtMs);
  }

  async delete(id: string): Promise<void> {
    const database = await this.open();
    await run(database, 'readwrite', (store) => store.delete(id));
  }

  private open(): Promise<IDBDatabase> {
    // Cached so concurrent calls share one connection rather than racing to upgrade.
    this.database ??= new Promise((resolve, reject) => {
      const request = indexedDB.open(DATABASE, VERSION);
      request.onupgradeneeded = () => {
        if (!request.result.objectStoreNames.contains(STORE)) {
          request.result.createObjectStore(STORE, { keyPath: 'id' });
        }
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    return this.database;
  }
}

function run<T>(
  database: IDBDatabase,
  mode: IDBTransactionMode,
  operation: (store: IDBObjectStore) => IDBRequest,
): Promise<T> {
  return new Promise((resolve, reject) => {
    const transaction = database.transaction(STORE, mode);
    const request = operation(transaction.objectStore(STORE));
    request.onsuccess = () => resolve(request.result as T);
    // Both handlers matter: a request can succeed and its transaction still abort on commit.
    request.onerror = () => reject(request.error);
    transaction.onabort = () => reject(transaction.error);
  });
}
