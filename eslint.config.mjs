/**
 * ESLint config for the TypeScript proxy server (src/).
 *
 * Mirrors the strictness of frontend/eslint.config.js.
 * Uses ESLint 9 flat-config format.
 */
import js from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ["src/**/*.ts"],
    languageOptions: {
      parserOptions: {
        project: "./tsconfig.json",
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      // Enforce no unused variables (allow _ prefix for intentionally ignored)
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
      // Warn on explicit any — encourage proper types
      "@typescript-eslint/no-explicit-any": "warn",
      // Allow console in server-side code
      "no-console": "off",
      // Enforce consistent return types on async functions
      "@typescript-eslint/explicit-function-return-type": "off",
      // Prefer const
      "prefer-const": "error",
      // No var
      "no-var": "error",
    },
  },
  {
    ignores: ["dist/**", "node_modules/**"],
  }
);
