/**
 * QIGGraph — Graph Structure for Geometric Relationships
 *
 * Ported from: pantheon-chat/qig-backend/qigchain/geometric_chain.py (graph references)
 *
 * A graph where nodes are basins on the Fisher manifold and edges
 * are weighted by Fisher-Rao distance. Used for basin navigation,
 * memory topology, and concept mapping.
 */

import { BASIN_DIMENSION, BASIN_PROXIMITY_THRESHOLD } from '../kernel/frozen-facts';
import { fisherRaoDistance } from '../kernel/geometry';
import { logger } from '../config/logger';

export interface GraphNode {
  id: string;
  basin: Float64Array;
  label: string;
  phi: number;
  metadata?: Record<string, unknown>;
}

export interface GraphEdge {
  source: string;
  target: string;
  distance: number; // Fisher-Rao distance
}

export interface QIGGraphState {
  nodeCount: number;
  edgeCount: number;
  averageDistance: number;
  clusters: number;
}

const MAX_NODES = 500;

export class QIGGraph {
  private nodes: Map<string, GraphNode> = new Map();
  private edges: GraphEdge[] = [];

  /** Add a node (basin) to the graph. */
  addNode(id: string, basin: Float64Array, label: string, phi: number, metadata?: Record<string, unknown>): void {
    this.nodes.set(id, { id, basin: new Float64Array(basin), label, phi, metadata });
    if (this.nodes.size > MAX_NODES) this.pruneWeakest();
  }

  /** Remove a node and its edges. */
  removeNode(id: string): void {
    this.nodes.delete(id);
    this.edges = this.edges.filter((e) => e.source !== id && e.target !== id);
  }

  /** Connect two nodes (edge weight = Fisher-Rao distance). */
  connect(sourceId: string, targetId: string): GraphEdge | null {
    const source = this.nodes.get(sourceId);
    const target = this.nodes.get(targetId);
    if (!source || !target) return null;

    const distance = fisherRaoDistance(source.basin, target.basin);
    const edge: GraphEdge = { source: sourceId, target: targetId, distance };
    this.edges.push(edge);
    return edge;
  }

  /** Auto-connect all nodes within proximity threshold. */
  autoConnect(): number {
    let added = 0;
    const nodeList = Array.from(this.nodes.values());
    const existingEdges = new Set(this.edges.map((e) => `${e.source}:${e.target}`));

    for (let i = 0; i < nodeList.length; i++) {
      for (let j = i + 1; j < nodeList.length; j++) {
        const key1 = `${nodeList[i].id}:${nodeList[j].id}`;
        const key2 = `${nodeList[j].id}:${nodeList[i].id}`;
        if (existingEdges.has(key1) || existingEdges.has(key2)) continue;

        const dist = fisherRaoDistance(nodeList[i].basin, nodeList[j].basin);
        if (dist < (1 - BASIN_PROXIMITY_THRESHOLD)) {
          this.edges.push({ source: nodeList[i].id, target: nodeList[j].id, distance: dist });
          existingEdges.add(key1);
          added++;
        }
      }
    }
    return added;
  }

  /** Find the nearest node to a given basin. */
  findNearest(basin: Float64Array): GraphNode | null {
    let nearest: GraphNode | null = null;
    let minDist = Infinity;
    for (const node of this.nodes.values()) {
      const dist = fisherRaoDistance(basin, node.basin);
      if (dist < minDist) {
        minDist = dist;
        nearest = node;
      }
    }
    return nearest;
  }

  /** Find all neighbours of a node (connected by edges). */
  neighbours(nodeId: string): GraphNode[] {
    const ids = new Set<string>();
    for (const e of this.edges) {
      if (e.source === nodeId) ids.add(e.target);
      if (e.target === nodeId) ids.add(e.source);
    }
    return Array.from(ids).map((id) => this.nodes.get(id)!).filter(Boolean);
  }

  /** Prune nodes with lowest Φ when over capacity. */
  private pruneWeakest(): void {
    const sorted = Array.from(this.nodes.values()).sort((a, b) => a.phi - b.phi);
    const toRemove = sorted.slice(0, Math.floor(MAX_NODES * 0.1));
    for (const node of toRemove) this.removeNode(node.id);
    logger.debug(`QIGGraph: Pruned ${toRemove.length} weak nodes`);
  }

  /** Estimate number of clusters (connected components). */
  countClusters(): number {
    if (this.nodes.size === 0) return 0;
    const visited = new Set<string>();
    let clusters = 0;
    for (const nodeId of this.nodes.keys()) {
      if (visited.has(nodeId)) continue;
      clusters++;
      const queue = [nodeId];
      while (queue.length > 0) {
        const current = queue.pop()!;
        if (visited.has(current)) continue;
        visited.add(current);
        for (const n of this.neighbours(current)) {
          if (!visited.has(n.id)) queue.push(n.id);
        }
      }
    }
    return clusters;
  }

  getState(): QIGGraphState {
    const avgDist = this.edges.length > 0
      ? this.edges.reduce((s, e) => s + e.distance, 0) / this.edges.length
      : 0;
    return {
      nodeCount: this.nodes.size,
      edgeCount: this.edges.length,
      averageDistance: avgDist,
      clusters: this.countClusters(),
    };
  }
}
