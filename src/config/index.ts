/**
 * Vex Agent — Centralised Configuration
 * All env vars resolved once at startup.
 */

export const config = {
  port: parseInt(process.env.PORT || '8080', 10),
  nodeEnv: process.env.NODE_ENV || 'development',

  // Ollama (local LLM — primary)
  ollama: {
    url: process.env.OLLAMA_URL || 'http://ollama.railway.internal:11434',
    model: process.env.OLLAMA_MODEL || 'lfm2.5-thinking:1.2b',
    enabled: process.env.OLLAMA_ENABLED !== 'false', // enabled by default
    timeoutMs: parseInt(process.env.OLLAMA_TIMEOUT_MS || '120000', 10),
  },

  // External LLM (fallback)
  llm: {
    apiKey: process.env.LLM_API_KEY || '',
    baseUrl: process.env.LLM_BASE_URL || 'https://api.openai.com/v1',
    model: process.env.LLM_MODEL || 'gpt-4.1-mini',
  },

  // HuggingFace
  hfToken: process.env.HF_TOKEN || '',

  // Data persistence
  dataDir: process.env.DATA_DIR || './data/workspace',
  trainingDir: process.env.TRAINING_DIR || '/data/training',

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
