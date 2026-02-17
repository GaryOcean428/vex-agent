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
    timeoutMs: parseInt(process.env.OLLAMA_TIMEOUT_MS || '300000', 10),
  },

  // External LLM (fallback)
  llm: {
    apiKey: process.env.LLM_API_KEY || '',
    baseUrl: process.env.LLM_BASE_URL || 'https://api.openai.com/v1',
    model: process.env.LLM_MODEL || 'gpt-4.1-mini',
  },

  // HuggingFace
  hfToken: process.env.HF_TOKEN || '',

  // Additional LLM providers (mirrored from monkey1 shared vars)
  anthropicApiKey: process.env.ANTHROPIC_API_KEY || '',
  openaiApiKey: process.env.OPENAI_API_KEY || '',
  xaiApiKey: process.env.XAI_API_KEY || '',
  geminiApiKey: process.env.GEMINI_API_KEY || '',
  groqApiKey: process.env.GROQ_API_KEY || '',
  perplexityApiKey: process.env.PERPLEXITY_API_KEY || '',

  // Search / tools
  tavilyApiKey: process.env.TAVILY_API_KEY || '',

  // ComputeSDK (sandbox code execution)
  computeSdk: {
    url: process.env.COMPUTE_SDK_URL || 'http://computesdk.railway.internal:3000',
    apiKey: process.env.COMPUTE_SDK_API_KEY || '',
    enabled: process.env.COMPUTE_SDK_ENABLED !== 'false',
  },

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

  // Chat auth
  chatAuthToken: process.env.CHAT_AUTH_TOKEN || '',

  // Logging
  logLevel: process.env.LOG_LEVEL || 'info',
} as const;
