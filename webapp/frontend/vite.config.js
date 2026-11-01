import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// `npm run build` writes the page into webapp/static, which the FastAPI server serves at /.
// `npm run dev` serves on :5173 and forwards /api to the backend on :8000.
export default defineConfig({
  plugins: [vue()],
  build: { outDir: '../static', emptyOutDir: true, chunkSizeWarningLimit: 2000 },
  server: { proxy: { '/api': 'http://127.0.0.1:8000' } },
})
