/**
 * TypeScript Definitions for SoupLite npm
 * Author: Muhtar Jaksilikov
 */

export interface SoupLiteOptions {
  maxMemoryMb?: number;
}

export interface MemoryStats {
  totalMemoryGb: number;
  freeMemoryGb: number;
  withinSub2GbBudget: boolean;
  status: string;
}

export class SoupLite {
  constructor(options?: SoupLiteOptions);
  getMemoryStats(): MemoryStats;
  runCLI(args?: string[]): string;
}

export const version: string;
export const author: string;
