import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: Object.fromEntries(
      [
        '/state', '/telemetry', '/status', '/basin', '/kernels', '/health',
        '/enqueue', '/memory', '/graph', '/sleep', '/admin', '/training',
        '/chat', '/conversations', '/auth', '/governor', '/beta-attention',
      ].map((path) => [path, 'http://localhost:8080']),
    ),
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
})
