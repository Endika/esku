import type { CustomSign } from '@domain/recognition/entities/CustomSign';
import type { ICustomSignRepository } from '@domain/recognition/repositories/ICustomSignRepository';

/**
 * In-memory repository, used by tests so they exercise the real use cases against a real
 * implementation of the port rather than a mock with rehearsed answers.
 */
export class InMemoryCustomSignRepository implements ICustomSignRepository {
  private readonly signs = new Map<string, CustomSign>();

  async save(sign: CustomSign): Promise<void> {
    this.signs.set(sign.id, sign);
  }

  async findAll(): Promise<CustomSign[]> {
    return [...this.signs.values()].sort((a, b) => a.createdAtMs - b.createdAtMs);
  }

  async delete(id: string): Promise<void> {
    this.signs.delete(id);
  }
}
