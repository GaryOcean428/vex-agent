import { create } from "zustand";

/* ── Types ─────────────────────────────────────────────────────────── */

export type ToastType = "success" | "error" | "info" | "warning";

export interface Toast {
  id: string;
  type: ToastType;
  message: string;
  duration: number;
  /** Set to true when the dismiss animation starts */
  dismissing: boolean;
}

interface ToastState {
  toasts: Toast[];
  add: (type: ToastType, message: string, duration?: number) => void;
  dismiss: (id: string) => void;
  remove: (id: string) => void;
}

/* ── Store ─────────────────────────────────────────────────────────── */

let counter = 0;
const nextId = () => `toast-${++counter}-${Date.now()}`;

export const useToastStore = create<ToastState>((set) => ({
  toasts: [],

  add(type, message, duration = 4000) {
    const id = nextId();
    set((s) => ({
      toasts: [...s.toasts, { id, type, message, duration, dismissing: false }],
    }));
  },

  dismiss(id) {
    set((s) => ({
      toasts: s.toasts.map((t) =>
        t.id === id ? { ...t, dismissing: true } : t,
      ),
    }));
  },

  remove(id) {
    set((s) => ({
      toasts: s.toasts.filter((t) => t.id !== id),
    }));
  },
}));

/* ── Convenience helper ────────────────────────────────────────────── */

export const toast = {
  success: (message: string, duration?: number) =>
    useToastStore.getState().add("success", message, duration),
  error: (message: string, duration?: number) =>
    useToastStore.getState().add("error", message, duration),
  info: (message: string, duration?: number) =>
    useToastStore.getState().add("info", message, duration),
  warning: (message: string, duration?: number) =>
    useToastStore.getState().add("warning", message, duration),
};
