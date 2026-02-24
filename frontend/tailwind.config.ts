/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // ============================================
        // NEON ELECTRIC COLORS - Primary Palette
        // ============================================
        neon: {
          'electric-blue': '#2563eb',   // Primary, actions
          'electric-cyan': '#00cec9',    // Accents, borders
          'electric-indigo': '#4f46e5',  // Secondary actions
          'electric-purple': '#6c5ce7',  // Gradients, effects
          'electric-magenta': '#fd79a8', // Interactive
          'electric-pink': '#ec4899',    // Hover states
          'electric-coral': '#ff4757',   // Alerts, destructive
          'electric-orange': '#ff7675',  // Warnings
          'electric-yellow': '#fdcb6e',  // Info, secondary alerts
          'electric-green': '#22c55e',   // Success
          'electric-lavender': '#a29bfe', // Subtle accents
        },

        // ============================================
        // LIGHT THEME
        // ============================================
        light: {
          bg: {
            primary: '#fefefe',      // Off-white
            secondary: '#f8f9fa',    // Warm gray
            tertiary: '#f1f3f4',     // Slightly darker
            quaternary: '#e9ecef',   // Subtle depth
            accent: '#ffffff',       // Cards, dialogs
          },
          text: {
            primary: '#2d3436',      // Dark charcoal
            secondary: '#636e72',    // Medium gray
            tertiary: '#74b9ff',     // Muted
            quaternary: '#a4afb7',   // Very muted
          },
          border: '#e9ecef',         // Subtle lines
          hover: '#f1f3f4',          // Hover state
        },

        // ============================================
        // DARK THEME - Deep Navy with Neon Accents
        // ============================================
        dark: {
          bg: {
            primary: '#0a0e1a',      // Very dark navy
            secondary: '#1a1f2e',    // Slightly lighter
            tertiary: '#2c3447',     // Medium navy
            quaternary: '#3c4558',   // Hover depth
            accent: '#252b3d',       // Cards, dialogs
          },
          text: {
            primary: '#f8f9fa',      // Pure white
            secondary: '#adb5bd',    // Light gray
            tertiary: '#6c757d',     // Muted gray
            quaternary: '#495057',   // Very muted
          },
          border: '#495057',         // Subtle dark lines
          hover: '#3c4558',          // Hover state

          // Accent colors for dark theme
          accent: {
            primary: '#00cec9',      // Cyan highlight
            secondary: '#2563eb',    // Electric blue
            success: '#22c55e',      // Success green
            warning: '#fdcb6e',      // Warning yellow
            danger: '#ff4757',       // Danger coral
          }
        },

        // ============================================
        // SEMANTIC COLORS
        // ============================================
        status: {
          success: '#00b894',
          warning: '#fdcb6e',
          error: '#ff4757',
          info: '#00cec9',
        },

        // ============================================
        // CHAT COLORS (if applicable)
        // ============================================
        chat: {
          user: {
            light: '#667eea',        // User bubble light
            dark: '#764ba2',         // User bubble dark
          },
          agent: {
            light: '#f093fb',        // Agent bubble light
            dark: '#2563eb',         // Agent bubble dark (electric blue)
          },
          system: {
            light: '#ffeaa7',        // System light
            dark: '#fdcb6e',         // System dark
          }
        },
      },

      // ============================================
      // BACKGROUND IMAGES & GRADIENTS
      // ============================================
      backgroundImage: {
        'grid-light': 'url("data:image/svg+xml,%3csvg xmlns=\'http://www.w3.org/2000/svg\' viewBox=\'0 0 32 32\' width=\'32\' height=\'32\' fill=\'none\' stroke=\'rgb(0 0 0 / 0.02)\'%3e%3cpath d=\'M0 .5H31.5V32\'/%3e%3c/svg%3e")',
        'grid-dark': 'url("data:image/svg+xml,%3csvg xmlns=\'http://www.w3.org/2000/svg\' viewBox=\'0 0 32 32\' width=\'32\' height=\'32\' fill=\'none\' stroke=\'rgb(255 255 255 / 0.03)\'%3e%3cpath d=\'M0 .5H31.5V32\'/%3e%3c/svg%3e")',

        'gradient-brand': 'linear-gradient(135deg, #ff4757 0%, #ff7675 25%, #fdcb6e 50%, #00cec9 75%, #a29bfe 100%)',
        'gradient-electric': 'linear-gradient(135deg, #2563eb 0%, #00cec9 50%, #ec4899 100%)',
        'gradient-neon': 'linear-gradient(90deg, #00cec9 0%, #6c5ce7 50%, #ff4757 100%)',
        'gradient-chat-user': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'gradient-chat-agent': 'linear-gradient(135deg, #2563eb 0%, #ec4899 100%)',
        'gradient-neural': 'radial-gradient(circle at center, rgba(0, 206, 201, 0.1) 0%, transparent 50%)',
      },

      // ============================================
      // SHADOWS & GLOW EFFECTS
      // ============================================
      boxShadow: {
        // Light theme shadows
        'light-sm': '0 1px 2px rgba(0, 0, 0, 0.05)',
        'light-md': '0 4px 6px rgba(0, 0, 0, 0.1)',
        'light-lg': '0 10px 15px rgba(0, 0, 0, 0.1)',

        // Dark theme shadows
        'dark-sm': '0 1px 3px rgba(0, 0, 0, 0.3)',
        'dark-md': '0 4px 12px rgba(0, 0, 0, 0.3)',
        'dark-lg': '0 15px 30px rgba(0, 0, 0, 0.4)',

        // Neon glow effects
        'glow-electric-blue': '0 0 20px rgba(37, 99, 235, 0.4)',
        'glow-electric-cyan': '0 0 20px rgba(0, 206, 201, 0.4)',
        'glow-electric-purple': '0 0 20px rgba(108, 92, 231, 0.4)',
        'glow-electric-pink': '0 0 20px rgba(236, 72, 153, 0.4)',
        'glow-electric-coral': '0 0 20px rgba(255, 71, 87, 0.4)',
        'glow-electric-magenta': '0 0 20px rgba(253, 121, 168, 0.4)',

        // Chat shadows
        'chat-light': '0 2px 8px rgba(0, 0, 0, 0.08)',
        'chat-dark': '0 4px 12px rgba(0, 0, 0, 0.4)',
        'message-hover': '0 4px 16px rgba(0, 206, 201, 0.2)',
      },

      // ============================================
      // ANIMATIONS & KEYFRAMES
      // ============================================
      animation: {
        'pulse-soft': 'pulse-soft 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'typing': 'typing 1.5s infinite',
        'shimmer': 'shimmer 2s infinite',
        'float': 'float 3s ease-in-out infinite',
        'neon-pulse': 'neon-pulse 2s ease-in-out infinite',
      },

      keyframes: {
        'pulse-soft': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.7' },
        },
        'glow': {
          '0%': { boxShadow: '0 0 5px rgba(0, 206, 201, 0.2)' },
          '100%': { boxShadow: '0 0 20px rgba(0, 206, 201, 0.6)' },
        },
        'typing': {
          '0%, 60%': { opacity: '1' },
          '30%': { opacity: '0.4' },
        },
        'shimmer': {
          '0%': { backgroundPosition: '-1000px 0' },
          '100%': { backgroundPosition: '1000px 0' },
        },
        'float': {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        'neon-pulse': {
          '0%, 100%': { textShadow: '0 0 10px rgba(0, 206, 201, 0.3)' },
          '50%': { textShadow: '0 0 20px rgba(0, 206, 201, 0.8)' },
        },
      },

      // ============================================
      // TYPOGRAPHY
      // ============================================
      fontFamily: {
        'display': ['Inter', 'system-ui', 'sans-serif'],
        'body': ['Inter', 'system-ui', 'sans-serif'],
        'mono': ['JetBrains Mono', 'Fira Code', 'monospace'],
      },

      fontSize: {
        'xs': ['12px', { lineHeight: '16px' }],
        'sm': ['14px', { lineHeight: '20px' }],
        'base': ['16px', { lineHeight: '24px' }],
        'lg': ['18px', { lineHeight: '28px' }],
        'xl': ['20px', { lineHeight: '28px' }],
        '2xl': ['24px', { lineHeight: '32px' }],
      },
    },
  },
  plugins: [],
}
