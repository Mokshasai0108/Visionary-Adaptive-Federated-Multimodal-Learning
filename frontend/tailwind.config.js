/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        afspl: {
          bg: "#0a0f1e",
          surface: "#111827",
          card: "#1a2235",
          border: "#1e3a5f",
          accent: "#00d4ff",
          green: "#00ff9d",
          amber: "#ffb800",
          red: "#ff4757",
          muted: "#4a6280",
          text: "#e2eaf5",
        },
      },
      fontFamily: {
        mono: ["'JetBrains Mono'", "monospace"],
        sans: ["'DM Sans'", "sans-serif"],
        display: ["'Space Grotesk'", "sans-serif"],
      },
    },
  },
  plugins: [],
};
