import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // 开发模式：前端走 Vite 端口，API 转发给 pa serve
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
