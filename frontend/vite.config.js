import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

// GitHub Pages base path · production 빌드에서만 적용
// repo: heotaewoong/aws_say2_project → https://heotaewoong.github.io/aws_say2_project/
const REPO_BASE = '/aws_say2_project/';

// https://vitejs.dev/config/
export default defineConfig(({ command }) => ({
  base: command === 'build' ? REPO_BASE : '/',
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true, // 5173 점유 시 fallback 대신 에러 — SMART Launcher URL 일관성 유지
    open: true, // auto-open browser on `npm run dev`
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    // Multi-entry: SMART on FHIR launch/callback HTML 페이지를 별도 entry로 빌드.
    // index.html은 React SPA, launch.html/app.html은 fhirclient 단독 페이지.
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        launch: resolve(__dirname, 'launch.html'),
        app: resolve(__dirname, 'app.html'),
      },
    },
  },
}));
