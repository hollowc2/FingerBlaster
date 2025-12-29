import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  // Load env vars from project root (parent directory)
  const rootDir = path.resolve(__dirname, '..', '..');
  const env = loadEnv(mode, rootDir, '');
  
  return {
    server: {
      port: 3000,
      host: '0.0.0.0',
      proxy: {
        // Proxy API requests to FastAPI backend
        '/api': {
          target: 'http://localhost:8000',
          changeOrigin: true,
        },
        // Proxy WebSocket connections
        '/ws': {
          target: 'ws://localhost:8000',
          ws: true,
        },
      },
    },
    plugins: [react()],
    define: {
      'process.env.API_KEY': JSON.stringify(env.GEMINI_API_KEY),
      'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY),
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
  };
});
