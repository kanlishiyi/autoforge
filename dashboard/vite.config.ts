import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";

export default defineConfig({
  plugins: [react()],
  base: "/dashboard/",
  server: {
    port: 5173,
  },
  preview: {
    port: 4173,
  },
});
