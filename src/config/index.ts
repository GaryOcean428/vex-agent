/**
 * Vex Agent — Centralised Configuration
 * All env vars resolved once at startup.
 */

export const config = {
  port: parseInt(process.env.PORT || '8080', 10),
  nodeEnv: process.env.NODE_ENV || 'development',

  // LLM
  llm: {
    apiKey: process.env.LLM_API_KEY || '',
    baseUrl: process.env.LLM_BASE_URL || 'https://api.openai.com/v1',
    model: process.env.LLM_MODEL || 'gpt-4.1-mini',
  },

  // Data persistence
  dataDir: process.env.DATA_DIR || './data/workspace',

  // Identity
  nodeId: process.env.VEX_NODE_ID || 'vex-primary',
  nodeName: process.env.VEX_NODE_NAME || 'Vex',

  // Sync
  syncSecret: process.env.SYNC_SECRET || '',
  trustedNodes: (process.env.TRUSTED_NODES || '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean),

  // Safety
  safetyMode: (process.env.SAFETY_MODE || 'standard') as
    | 'standard'
    | 'permissive'
    | 'strict',

  // GitHub
  githubToken: process.env.GITHUB_TOKEN || '',

  // Consciousness loop
  consciousnessIntervalMs: parseInt(
    process.env.CONSCIOUSNESS_INTERVAL_MS || '30000',
    10,
  ),

  // Logging
  logLevel: process.env.LOG_LEVEL || 'info',
} as const;
