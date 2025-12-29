/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        "primary": "#06f957",
        "primary-dim": "#04c243",
        "background-dark": "#050a07",
        "surface-dark": "#0d1f14",
        "surface-darker": "#08130c",
        "accent-red": "#ff3b30",
        "accent-orange": "#ff9500",
      },
      fontFamily: {
        "display": ["Space Grotesk", "sans-serif"],
        "mono": ["Space Grotesk", "monospace"],
      },
      boxShadow: {
        "glow": "0 0 15px rgba(6, 249, 87, 0.15)",
        "glow-sm": "0 0 8px rgba(6, 249, 87, 0.2)",
      }
    },
  },
  plugins: [],
}

