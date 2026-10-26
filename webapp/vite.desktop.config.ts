import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

export default defineConfig({
  root: fileURLToPath(new URL("./desktop", import.meta.url)),
  plugins: [react()],
  define: { __HYDROSENSE_DESKTOP__: true },
  build: { outDir: "../desktop-dist", emptyOutDir: true, assetsDir: "assets" },
});
