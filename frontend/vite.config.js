import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// CloudFront + S3 배포용: base '/' (origin path /frontend 설정됨)
// GitHub Pages 사용 시에는 '/aws_say2_project/'로 변경 필요

// https://vitejs.dev/config/
export default defineConfig(({ command }) => ({
  base: '/',
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
