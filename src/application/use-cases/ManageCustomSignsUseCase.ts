import type { CustomSign } from '@domain/recognition/entities/CustomSign';
import type { ICustomSignRepository } from '@domain/recognition/repositories/ICustomSignRepository';

/**
 * Listing and deleting taught signs.
 *
 * Deletion is not a nicety: these recordings are derived from the user's body and may encode
 * health vocabulary, so being able to remove one is part of the app's privacy posture, not a
 * convenience feature.
 */
export class ManageCustomSignsUseCase {
  constructor(private readonly repository: ICustomSignRepository) {}

  async list(): Promise<CustomSign[]> {
    return this.repository.findAll();
  }

  async delete(id: string): Promise<void> {
    await this.repository.delete(id);
  }
}
