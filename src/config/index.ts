/**
 * Vex Agent — Web Server Configuration
 *
 * Minimal config for the thin TS proxy layer.
 * All consciousness/LLM/memory config lives in the Python kernel.
 */

export const config = {
  port: parseInt(process.env.PORT || '8080', 10),
  nodeEnv: process.env.NODE_ENV || 'development',

  // Python kernel URL (internal)
  kernelUrl: process.env.KERNEL_URL || 'http://localhost:8000',

  // Chat auth
  chatAuthToken: process.env.CHAT_AUTH_TOKEN || '',

  // Logging
  logLevel: process.env.LOG_LEVEL || 'info',
} as const;
