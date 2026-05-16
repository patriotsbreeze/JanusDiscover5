/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        janus: {
          50:  '#eef5ff',
          100: '#d9e8ff',
          200: '#bbd3ff',
          300: '#8ab6ff',
          400: '#5590ff',
          500: '#2C6BFF',
          600: '#1a4ef5',
          700: '#1339e0',
          800: '#162fb6',
          900: '#182d8f',
          950: '#131f5c',
        },
        bio: {
          green: '#1A9641',
          teal:  '#2CA25F',
          gold:  '#FDAE61',
          red:   '#D7191C',
        },
      },
      fontFamily: {
        sans: ['Inter var', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4,0,0.6,1) infinite',
        'spin-slow':  'spin 3s linear infinite',
      },
    },
  },
  plugins: [],
}
