import { useEffect } from "react";
import { useToastStore } from "../stores/toastStore.ts";
import type { Toast } from "../stores/toastStore.ts";
import "./Toast.css";

/* ── Icons ─────────────────────────────────────────────────────────── */

const ICONS: Record<Toast["type"], string> = {
  success: "\u2713",
  error: "\u2717",
  warning: "\u26A0",
  info: "\u2139",
};

/* ── Single Toast ──────────────────────────────────────────────────── */

function ToastItem({ id, type, message, duration, dismissing }: Toast) {
  const { dismiss, remove } = useToastStore();

  // Auto-dismiss after `duration` ms
  useEffect(() => {
    const timer = setTimeout(() => dismiss(id), duration);
    return () => clearTimeout(timer);
  }, [id, duration, dismiss]);

  // After the fade-out animation finishes, remove from the store
  useEffect(() => {
    if (!dismissing) return;
    const timer = setTimeout(() => remove(id), 300); // matches CSS animation
    return () => clearTimeout(timer);
  }, [dismissing, id, remove]);

  const className = [
    "toast",
    `toast--${type}`,
    dismissing ? "toast--dismissing" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={className} role="alert">
      <span className="toast__icon">{ICONS[type]}</span>
      <span className="toast__message">{message}</span>
      <button
        className="toast__close"
        onClick={() => dismiss(id)}
        aria-label="Dismiss notification"
      >
        &times;
      </button>
    </div>
  );
}

/* ── Container ─────────────────────────────────────────────────────── */

export function ToastContainer() {
  const toasts = useToastStore((s) => s.toasts);

  if (toasts.length === 0) return null;

  return (
    <div className="toast-container" aria-live="polite">
      {toasts.map((t) => (
        <ToastItem key={t.id} {...t} />
      ))}
    </div>
  );
}
