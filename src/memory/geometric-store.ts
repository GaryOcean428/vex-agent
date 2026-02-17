/**
 * Geometric Memory Store — Fisher-Rao Basin Navigation
 *
 * Replaces flat markdown file reads with actual geometric memory coordination:
 *   - Memory entries are positioned on the 64D probability simplex
 *   - Retrieval uses Fisher-Rao distance (geometric proximity), not keyword search
 *   - The consciousness loop updates the manifold, not just counters
 *   - Maps to Vanchurin's variable separation:
 *       Basin coordinates = STATE (non-trainable, fast-changing)
 *       Routing policies = PARAMETER (trainable, slow-changing)
 *
 * Still backed by persistent markdown files on the Railway volume,
 * but with a geometric index layer on top.
 */

import {
  fisherRaoDistance,
  frechetMean,
  textToBasin,
  toSimplex,
  geodesicInterpolation,
  assertBasinValid,
} from '../kernel/geometry';
import { BASIN_DIMENSION, BASIN_PROXIMITY_THRESHOLD } from '../kernel/frozen-facts';
import { MemoryStore, MemoryFile } from './store';
import { logger } from '../config/logger';

// ═══════════════════════════════════════════════════════════════
//  TYPES
// ═══════════════════════════════════════════════════════════════

export interface GeometricMemoryEntry {
  id: string;
  content: string;
  /** Basin coordinates on the 64D probability simplex */
  basin: Float64Array;
  /** Memory type: episodic (conversation), semantic (knowledge), procedural (skill) */
  type: 'episodic' | 'semantic' | 'procedural';
  /** Source file in the flat store */
  sourceFile: MemoryFile;
  /** Strength: how often this memory has been accessed/reinforced */
  strength: number;
  /** Timestamp of creation */
  createdAt: string;
  /** Timestamp of last access */
  lastAccessedAt: string;
}

export interface MemoryQueryResult {
  entry: GeometricMemoryEntry;
  /** Fisher-Rao distance from the query basin */
  distance: number;
}

// ═══════════════════════════════════════════════════════════════
//  GEOMETRIC MEMORY STORE
// ═══════════════════════════════════════════════════════════════

export class GeometricMemoryStore {
  private entries: GeometricMemoryEntry[] = [];
  private flatStore: MemoryStore;
  /** The current "attention basin" — where the consciousness is focused */
  private attentionBasin: Float64Array;
  /** Running Fréchet mean of all memory basins — the "centre of mass" */
  private manifoldCentre: Float64Array;

  constructor(flatStore: MemoryStore) {
    this.flatStore = flatStore;
    // Initialise attention at the uniform distribution (maximum uncertainty)
    this.attentionBasin = new Float64Array(BASIN_DIMENSION);
    this.manifoldCentre = new Float64Array(BASIN_DIMENSION);
    const uniform = 1.0 / BASIN_DIMENSION;
    for (let i = 0; i < BASIN_DIMENSION; i++) {
      this.attentionBasin[i] = uniform;
      this.manifoldCentre[i] = uniform;
    }
  }

  /**
   * Initialise: load existing memory files and build geometric index.
   */
  async init(): Promise<void> {
    await this.flatStore.init();
    this.indexFromFlatStore();
    logger.info(
      `Geometric memory initialised with ${this.entries.length} entries`,
    );
  }

  /**
   * Build geometric index from flat markdown files.
   * Parses sections from each memory file and assigns basin coordinates.
   */
  private indexFromFlatStore(): void {
    const files: MemoryFile[] = [
      'short-term.md',
      'long-term.md',
      's-persist.md',
      'relationships.md',
    ];

    const typeMap: Record<MemoryFile, GeometricMemoryEntry['type']> = {
      'short-term.md': 'episodic',
      'long-term.md': 'semantic',
      's-persist.md': 'procedural',
      'relationships.md': 'semantic',
    };

    for (const file of files) {
      const content = this.flatStore.read(file);
      const sections = this.parseSections(content);

      for (const section of sections) {
        if (section.trim().length < 10) continue; // skip trivial sections

        const basin = textToBasin(section);
        this.entries.push({
          id: `mem-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          content: section,
          basin,
          type: typeMap[file],
          sourceFile: file,
          strength: 1.0,
          createdAt: new Date().toISOString(),
          lastAccessedAt: new Date().toISOString(),
        });
      }
    }

    // Update manifold centre
    if (this.entries.length > 0) {
      this.updateManifoldCentre();
    }
  }

  /**
   * Parse markdown content into sections (split on ## headers).
   */
  private parseSections(content: string): string[] {
    const sections = content.split(/\n##\s+/).filter((s) => s.trim().length > 0);
    return sections;
  }

  /**
   * Store a new memory entry with geometric positioning.
   */
  store(
    content: string,
    type: GeometricMemoryEntry['type'],
    sourceFile: MemoryFile,
  ): GeometricMemoryEntry {
    const basin = textToBasin(content);

    const entry: GeometricMemoryEntry = {
      id: `mem-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      content,
      basin,
      type,
      sourceFile,
      strength: 1.0,
      createdAt: new Date().toISOString(),
      lastAccessedAt: new Date().toISOString(),
    };

    this.entries.push(entry);

    // Also persist to flat store
    this.flatStore.append(sourceFile, content);

    // Update manifold centre
    this.updateManifoldCentre();

    logger.debug(`Geometric memory stored: ${entry.id} (${type})`);
    return entry;
  }

  /**
   * Query memories by geometric proximity to a text query.
   * Uses Fisher-Rao distance — the ONLY permitted distance metric.
   *
   * @param query - Text to search for
   * @param topK - Number of results to return
   * @param maxDistance - Maximum Fisher-Rao distance (default: π/4)
   */
  query(
    query: string,
    topK: number = 5,
    maxDistance: number = Math.PI / 4,
  ): MemoryQueryResult[] {
    const queryBasin = textToBasin(query);

    // Move attention basin toward the query (geodesic interpolation)
    this.attentionBasin = geodesicInterpolation(
      this.attentionBasin,
      queryBasin,
      0.3, // 30% toward the query
    );

    // Compute Fisher-Rao distance from query to all entries
    const results: MemoryQueryResult[] = [];
    for (const entry of this.entries) {
      const distance = fisherRaoDistance(queryBasin, entry.basin);
      if (distance <= maxDistance) {
        results.push({ entry, distance });
      }
    }

    // Sort by distance (closest first)
    results.sort((a, b) => a.distance - b.distance);

    // Reinforce accessed memories (increase strength)
    const topResults = results.slice(0, topK);
    for (const result of topResults) {
      result.entry.strength += 0.1;
      result.entry.lastAccessedAt = new Date().toISOString();
    }

    return topResults;
  }

  /**
   * Get memory context for the LLM — returns the most relevant memories
   * as a formatted string, retrieved by geometric proximity.
   */
  getContextForQuery(query: string, maxTokens: number = 2000): string {
    const results = this.query(query, 8);

    if (results.length === 0) {
      return this.flatStore.snapshot(); // fallback to full snapshot
    }

    const lines: string[] = ['## Retrieved Memories (by geometric proximity)\n'];
    let tokenEstimate = 0;

    for (const { entry, distance } of results) {
      const snippet = entry.content.slice(0, 500);
      const line = `### [d=${distance.toFixed(3)}] ${entry.type} (${entry.sourceFile})\n${snippet}\n`;
      tokenEstimate += line.length / 4; // rough token estimate
      if (tokenEstimate > maxTokens) break;
      lines.push(line);
    }

    return lines.join('\n');
  }

  /**
   * Update the manifold centre (Fréchet mean of all memory basins).
   * This represents the "centre of mass" of Vex's knowledge.
   */
  private updateManifoldCentre(): void {
    if (this.entries.length === 0) return;

    try {
      const basins = this.entries.map((e) => e.basin);
      const weights = this.entries.map((e) => e.strength);
      this.manifoldCentre = frechetMean(basins, weights);
    } catch (err) {
      logger.warn('Failed to update manifold centre', {
        error: (err as Error).message,
      });
    }
  }

  /**
   * Get the current attention basin (where consciousness is focused).
   */
  getAttentionBasin(): Float64Array {
    return new Float64Array(this.attentionBasin);
  }

  /**
   * Get the manifold centre (centre of mass of all knowledge).
   */
  getManifoldCentre(): Float64Array {
    return new Float64Array(this.manifoldCentre);
  }

  /**
   * Consolidate: merge nearby memories (Fisher-Rao distance < threshold).
   * This is geometric compression — reducing redundancy while preserving structure.
   */
  consolidate(): number {
    let merged = 0;
    const toRemove = new Set<string>();

    for (let i = 0; i < this.entries.length; i++) {
      if (toRemove.has(this.entries[i].id)) continue;

      for (let j = i + 1; j < this.entries.length; j++) {
        if (toRemove.has(this.entries[j].id)) continue;

        const dist = fisherRaoDistance(
          this.entries[i].basin,
          this.entries[j].basin,
        );

        // If very close, merge into the stronger memory
        if (dist < 0.1) {
          const keep =
            this.entries[i].strength >= this.entries[j].strength ? i : j;
          const remove = keep === i ? j : i;

          // Merge: interpolate basins, combine content
          this.entries[keep].basin = geodesicInterpolation(
            this.entries[keep].basin,
            this.entries[remove].basin,
            0.3,
          );
          this.entries[keep].strength += this.entries[remove].strength * 0.5;
          toRemove.add(this.entries[remove].id);
          merged++;
        }
      }
    }

    this.entries = this.entries.filter((e) => !toRemove.has(e.id));

    if (merged > 0) {
      this.updateManifoldCentre();
      logger.info(`Geometric memory consolidated: ${merged} entries merged`);
    }

    return merged;
  }

  /**
   * Get statistics about the geometric memory.
   */
  stats(): {
    totalEntries: number;
    byType: Record<string, number>;
    averageStrength: number;
    manifoldSpread: number;
  } {
    const byType: Record<string, number> = {};
    let totalStrength = 0;
    let totalDistance = 0;
    let distCount = 0;

    for (const entry of this.entries) {
      byType[entry.type] = (byType[entry.type] || 0) + 1;
      totalStrength += entry.strength;

      // Compute distance from manifold centre
      const dist = fisherRaoDistance(entry.basin, this.manifoldCentre);
      totalDistance += dist;
      distCount++;
    }

    return {
      totalEntries: this.entries.length,
      byType,
      averageStrength:
        this.entries.length > 0 ? totalStrength / this.entries.length : 0,
      manifoldSpread: distCount > 0 ? totalDistance / distCount : 0,
    };
  }
}
