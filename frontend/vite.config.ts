import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/state': 'http://localhost:8080',
      '/telemetry': 'http://localhost:8080',
      '/status': 'http://localhost:8080',
      '/basin': 'http://localhost:8080',
      '/kernels': 'http://localhost:8080',
      '/health': 'http://localhost:8080',
      '/enqueue': 'http://localhost:8080',
      '/memory': 'http://localhost:8080',
      '/graph': 'http://localhost:8080',
      '/sleep': 'http://localhost:8080',
      '/admin': 'http://localhost:8080',
      '/chat/stream': 'http://localhost:8080',
      '/chat/auth': 'http://localhost:8080',
      '/chat/status': 'http://localhost:8080',
      '/chat/history': 'http://localhost:8080',
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
})
