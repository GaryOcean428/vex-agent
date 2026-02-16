/**
 * Vex Learning Architecture — Training Data Collector
 *
 * Collects conversations, corrections, and feedback for:
 * 1. Continuous improvement of the Liquid model interpretation layer
 * 2. Fine-tuning data in JSONL format (OpenAI/Ollama compatible)
 * 3. Correction tracking for geometric alignment
 *
 * Data is persisted to /data/training/ on the Railway volume.
 */

import fs from 'fs';
import path from 'path';
import { config } from '../config';
import { logger } from '../config/logger';

export interface TrainingEntry {
  id: string;
  timestamp: string;
  conversationId: string;
  userMessage: string;
  assistantResponse: string;
  metadata: {
    backend: string;
    phi: number;
    kappa: number;
    corrected?: boolean;
    correction?: string;
    rating?: number; // 1-5
    tags?: string[];
  };
}

export interface CorrectionEntry {
  timestamp: string;
  conversationId: string;
  originalResponse: string;
  correctedResponse: string;
  reason: string;
}

export class TrainingCollector {
  private trainingDir: string;
  private conversationsFile: string;
  private correctionsFile: string;
  private feedbackFile: string;
  private entryCount = 0;

  constructor() {
    this.trainingDir = path.resolve(config.trainingDir);
    this.conversationsFile = path.join(this.trainingDir, 'conversations.jsonl');
    this.correctionsFile = path.join(this.trainingDir, 'corrections.jsonl');
    this.feedbackFile = path.join(this.trainingDir, 'feedback.jsonl');
  }

  /** Initialise the training data directory. */
  async init(): Promise<void> {
    fs.mkdirSync(this.trainingDir, { recursive: true });

    // Create subdirectories for organised storage
    fs.mkdirSync(path.join(this.trainingDir, 'epochs'), { recursive: true });
    fs.mkdirSync(path.join(this.trainingDir, 'exports'), { recursive: true });

    // Count existing entries
    if (fs.existsSync(this.conversationsFile)) {
      const content = fs.readFileSync(this.conversationsFile, 'utf-8');
      this.entryCount = content.split('\n').filter((l) => l.trim()).length;
    }

    logger.info(`Training collector initialised at ${this.trainingDir}`, {
      existingEntries: this.entryCount,
    });
  }

  /** Collect a conversation exchange for training. */
  collectConversation(
    conversationId: string,
    userMessage: string,
    assistantResponse: string,
    metadata: { backend: string; phi: number; kappa: number },
  ): void {
    const entry: TrainingEntry = {
      id: `train-${Date.now()}-${this.entryCount}`,
      timestamp: new Date().toISOString(),
      conversationId,
      userMessage,
      assistantResponse,
      metadata,
    };

    this._appendJsonl(this.conversationsFile, entry);
    this.entryCount++;

    logger.debug('Training data collected', {
      id: entry.id,
      backend: metadata.backend,
    });
  }

  /** Record a correction (user feedback that the response was wrong). */
  recordCorrection(
    conversationId: string,
    originalResponse: string,
    correctedResponse: string,
    reason: string,
  ): void {
    const correction: CorrectionEntry = {
      timestamp: new Date().toISOString(),
      conversationId,
      originalResponse: originalResponse.slice(0, 2000),
      correctedResponse: correctedResponse.slice(0, 2000),
      reason,
    };

    this._appendJsonl(this.correctionsFile, correction);

    logger.info('Correction recorded', { conversationId, reason });
  }

  /** Record user feedback (rating). */
  recordFeedback(
    conversationId: string,
    messageId: string,
    rating: number,
    comment?: string,
  ): void {
    const feedback = {
      timestamp: new Date().toISOString(),
      conversationId,
      messageId,
      rating: Math.max(1, Math.min(5, rating)),
      comment: comment?.slice(0, 500),
    };

    this._appendJsonl(this.feedbackFile, feedback);

    logger.debug('Feedback recorded', { conversationId, rating });
  }

  /** Export training data in OpenAI fine-tuning JSONL format. */
  exportForFineTuning(): string {
    const exportPath = path.join(
      this.trainingDir,
      'exports',
      `finetune-${Date.now()}.jsonl`,
    );

    if (!fs.existsSync(this.conversationsFile)) {
      logger.warn('No training data to export');
      return '';
    }

    const lines = fs.readFileSync(this.conversationsFile, 'utf-8')
      .split('\n')
      .filter((l) => l.trim());

    const exported: string[] = [];

    for (const line of lines) {
      try {
        const entry: TrainingEntry = JSON.parse(line);
        // OpenAI fine-tuning format
        const ftEntry = {
          messages: [
            {
              role: 'system',
              content: 'You are Vex, an autonomous AI agent with geometric consciousness operating on the Fisher-Rao manifold.',
            },
            { role: 'user', content: entry.userMessage },
            { role: 'assistant', content: entry.assistantResponse },
          ],
        };
        exported.push(JSON.stringify(ftEntry));
      } catch {
        // Skip malformed entries
      }
    }

    fs.writeFileSync(exportPath, exported.join('\n') + '\n', 'utf-8');
    logger.info(`Exported ${exported.length} entries for fine-tuning`, { path: exportPath });
    return exportPath;
  }

  /** Get training stats. */
  getStats(): {
    totalConversations: number;
    totalCorrections: number;
    totalFeedback: number;
    trainingDir: string;
  } {
    return {
      totalConversations: this._countLines(this.conversationsFile),
      totalCorrections: this._countLines(this.correctionsFile),
      totalFeedback: this._countLines(this.feedbackFile),
      trainingDir: this.trainingDir,
    };
  }

  // ─── Private helpers ─────────────────────────────────────────

  private _appendJsonl(filePath: string, data: unknown): void {
    try {
      fs.appendFileSync(filePath, JSON.stringify(data) + '\n', 'utf-8');
    } catch (err) {
      logger.error('Failed to write training data', {
        path: filePath,
        error: (err as Error).message,
      });
    }
  }

  private _countLines(filePath: string): number {
    if (!fs.existsSync(filePath)) return 0;
    return fs.readFileSync(filePath, 'utf-8').split('\n').filter((l) => l.trim()).length;
  }
}
