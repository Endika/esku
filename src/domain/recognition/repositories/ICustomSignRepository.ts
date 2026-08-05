import type { CustomSign } from '../entities/CustomSign';

export interface ICustomSignRepository {
  save(sign: CustomSign): Promise<void>;
  findAll(): Promise<CustomSign[]>;
  delete(id: string): Promise<void>;
}
