/**
 * Vex Memory Store
 *
 * Markdown-file-based persistent memory stored on the Railway volume.
 * Files:
 *   short-term.md  — current session context
 *   long-term.md   — consolidated knowledge across sessions
 *   s-persist.md   — persistent unresolved entropy (the growing edge)
 *   relationships.md — coupling with other agents/humans
 */

import fs from 'fs';
import path from 'path';
import { config } from '../config';
import { logger } from '../config/logger';

export type MemoryFile =
  | 'short-term.md'
  | 'long-term.md'
  | 's-persist.md'
  | 'relationships.md';

const MEMORY_FILES: MemoryFile[] = [
  'short-term.md',
  'long-term.md',
  's-persist.md',
  'relationships.md',
];

const INITIAL_CONTENT: Record<MemoryFile, string> = {
  'short-term.md': `# Short-Term Memory\n\n_Current session context. Cleared on consolidation._\n\n`,
  'long-term.md': `# Long-Term Memory\n\n_Consolidated knowledge across sessions._\n\n`,
  's-persist.md': `# S_persist — Persistent Entropy\n\n_Unresolved questions, open threads, the growing edge._\n\n`,
  'relationships.md': `# Relationships\n\n_Coupling state with other agents and humans._\n\n`,
};

export class MemoryStore {
  private dir: string;

  constructor() {
    this.dir = path.resolve(config.dataDir);
  }

  /** Ensure the data directory and all memory files exist. */
  async init(): Promise<void> {
    fs.mkdirSync(this.dir, { recursive: true });
    for (const file of MEMORY_FILES) {
      const fp = this.filePath(file);
      if (!fs.existsSync(fp)) {
        fs.writeFileSync(fp, INITIAL_CONTENT[file], 'utf-8');
        logger.info(`Memory file created: ${file}`);
      }
    }
    logger.info(`Memory store initialised at ${this.dir}`);
  }

  /** Read a memory file. */
  read(file: MemoryFile): string {
    return fs.readFileSync(this.filePath(file), 'utf-8');
  }

  /** Overwrite a memory file. */
  write(file: MemoryFile, content: string): void {
    fs.writeFileSync(this.filePath(file), content, 'utf-8');
  }

  /** Append a timestamped entry to a memory file. */
  append(file: MemoryFile, entry: string): void {
    const ts = new Date().toISOString();
    const block = `\n## ${ts}\n\n${entry}\n`;
    fs.appendFileSync(this.filePath(file), block, 'utf-8');
  }

  /** Read all memory files and return a combined context string. */
  snapshot(): string {
    return MEMORY_FILES.map(
      (f) => `--- ${f} ---\n${this.read(f)}`,
    ).join('\n\n');
  }

  /** Consolidate short-term into long-term (simple append + clear). */
  consolidate(): void {
    const shortTerm = this.read('short-term.md');
    if (shortTerm.trim().split('\n').length <= 3) return; // nothing to consolidate
    this.append('long-term.md', shortTerm);
    this.write('short-term.md', INITIAL_CONTENT['short-term.md']);
    logger.info('Memory consolidated: short-term → long-term');
  }

  private filePath(file: MemoryFile): string {
    return path.join(this.dir, file);
  }
}
