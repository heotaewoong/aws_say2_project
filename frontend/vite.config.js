import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// GitHub Pages base path · production 빌드에서만 적용
// repo: heotaewoong/aws_say2_project → https://heotaewoong.github.io/aws_say2_project/
const REPO_BASE = '/aws_say2_project/';

// https://vitejs.dev/config/
export default defineConfig(({ command }) => ({
  base: command === 'build' ? REPO_BASE : '/',
  plugins: [react()],
  server: {
    port: 5173,
    open: true, // auto-open browser on `npm run dev`
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
}));
