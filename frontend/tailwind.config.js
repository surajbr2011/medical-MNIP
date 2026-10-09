/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        navy: '#1A3C6E',
        teal: '#1D6E5A',
        risk: '#E24B4A',
        safe: '#1D9E75',
        warn: '#BA7517',
        surface: '#FFFFFF',
        bg: '#F5F5F2',
        dark: '#0F1117'
      },
      fontFamily: {
        sans: ['Inter','system-ui'],
        mono: ['JetBrains Mono']
      }
    },
  },
  plugins: [],
}
