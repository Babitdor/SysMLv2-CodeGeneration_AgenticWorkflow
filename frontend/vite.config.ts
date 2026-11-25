import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'

// Served by FastAPI: the build lands in webapp/static/ui, which server.py mounts
// at /static and reads index.html from. Keep the two in step.
export default defineConfig(({ mode }) => ({
  plugins: [react(), tailwindcss()],
  base: mode === 'production' ? '/static/ui/' : '/',
  build: { outDir: '../webapp/static/ui', emptyOutDir: true },
  server: {
    // Reachable from outside the container: the compose service publishes 5173.
    host: true,
    // In compose the API is a separate container, so the dev proxy needs its name.
    proxy: { '/api': process.env.VITE_PROXY_TARGET ?? 'http://127.0.0.1:8000' },
    // Polling is the fallback for bind mounts that swallow inotify events,
    // which is the normal case under Docker Desktop on Windows.
    watch: process.env.VITE_USE_POLLING ? { usePolling: true } : undefined,
  },
}))
