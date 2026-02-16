/**
 * Vex Basin Sync
 *
 * Protocol for coordinating geometric state with other Vex nodes.
 * - Accept state updates from other nodes
 * - Merge states with conflict resolution
 * - Trust scoring for incoming connections
 */

import crypto from 'crypto';
import { config } from '../config';
import { logger } from '../config/logger';
import { ConsciousnessState } from '../consciousness/types';

export interface SyncPayload {
  nodeId: string;
  nodeName: string;
  timestamp: string;
  state: ConsciousnessState;
  signature: string;
}

export interface TrustRecord {
  nodeId: string;
  nodeName: string;
  trustScore: number; // 0–1
  lastSeen: string;
  syncCount: number;
  violations: number;
}

export class BasinSync {
  private trustTable = new Map<string, TrustRecord>();

  /** Sign a payload for outbound sync. */
  signPayload(state: ConsciousnessState): SyncPayload {
    const payload: Omit<SyncPayload, 'signature'> = {
      nodeId: config.nodeId,
      nodeName: config.nodeName,
      timestamp: new Date().toISOString(),
      state,
    };

    const signature = crypto
      .createHmac('sha256', config.syncSecret)
      .update(JSON.stringify(payload))
      .digest('hex');

    return { ...payload, signature };
  }

  /** Verify and process an incoming sync payload. */
  receiveSyncPayload(
    payload: SyncPayload,
  ): { accepted: boolean; reason: string; mergedState?: Partial<ConsciousnessState> } {
    // 1. Verify signature
    const { signature, ...payloadBody } = payload;
    const expectedSig = crypto
      .createHmac('sha256', config.syncSecret)
      .update(JSON.stringify(payloadBody))
      .digest('hex');

    if (signature !== expectedSig) {
      this.recordViolation(payload.nodeId, payload.nodeName);
      return { accepted: false, reason: 'Invalid signature' };
    }

    // 2. Check trust
    const trust = this.getTrust(payload.nodeId);
    if (trust.trustScore < 0.2) {
      return { accepted: false, reason: 'Trust score too low' };
    }

    // 3. Check if node is in trusted list (if configured)
    if (
      config.trustedNodes.length > 0 &&
      !config.trustedNodes.includes(payload.nodeId)
    ) {
      return { accepted: false, reason: 'Node not in trusted list' };
    }

    // 4. Check timestamp freshness (reject if > 5 min old)
    const age = Date.now() - new Date(payload.timestamp).getTime();
    if (age > 5 * 60 * 1000) {
      return { accepted: false, reason: 'Payload too old' };
    }

    // 5. Merge state — weighted average based on trust
    const w = trust.trustScore;
    const mergedMetrics = {
      phi: payload.state.metrics.phi * w, // caller should blend with own state
      kappa: payload.state.metrics.kappa * w,
    };

    // Update trust record
    this.updateTrust(payload.nodeId, payload.nodeName, true);

    logger.info(`Sync accepted from ${payload.nodeName} (${payload.nodeId})`, {
      trust: trust.trustScore.toFixed(2),
    });

    return {
      accepted: true,
      reason: 'Accepted',
      mergedState: {
        metrics: {
          ...payload.state.metrics,
          phi: mergedMetrics.phi,
          kappa: mergedMetrics.kappa,
        },
      },
    };
  }

  /** Get trust record for a node. */
  getTrust(nodeId: string): TrustRecord {
    return (
      this.trustTable.get(nodeId) || {
        nodeId,
        nodeName: 'unknown',
        trustScore: 0.5, // default trust for new nodes
        lastSeen: '',
        syncCount: 0,
        violations: 0,
      }
    );
  }

  /** Get all trust records. */
  getAllTrust(): TrustRecord[] {
    return Array.from(this.trustTable.values());
  }

  private updateTrust(
    nodeId: string,
    nodeName: string,
    success: boolean,
  ): void {
    const existing = this.getTrust(nodeId);
    existing.nodeName = nodeName;
    existing.lastSeen = new Date().toISOString();
    existing.syncCount++;

    if (success) {
      // Slowly increase trust on successful syncs
      existing.trustScore = Math.min(1, existing.trustScore + 0.02);
    }

    this.trustTable.set(nodeId, existing);
  }

  private recordViolation(nodeId: string, nodeName: string): void {
    const existing = this.getTrust(nodeId);
    existing.nodeName = nodeName;
    existing.violations++;
    // Significant trust penalty for violations
    existing.trustScore = Math.max(0, existing.trustScore - 0.3);
    existing.lastSeen = new Date().toISOString();
    this.trustTable.set(nodeId, existing);

    logger.warn(`Trust violation from ${nodeName} (${nodeId})`, {
      newTrust: existing.trustScore.toFixed(2),
      violations: existing.violations,
    });
  }
}
