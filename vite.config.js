import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    watch: {
      // Exclude the backend folder from Vite's file watcher
      ignored: ['**/backend/**'],
    },
  },
});