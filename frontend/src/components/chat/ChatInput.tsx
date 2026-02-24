import { useCallback, useEffect, type KeyboardEvent, type RefObject } from "react";

interface ChatInputProps {
  input: string;
  isStreaming: boolean;
  inputRef: RefObject<HTMLTextAreaElement | null>;
  onChange: (value: string) => void;
  onKeyDown: (e: KeyboardEvent<HTMLTextAreaElement>) => void;
  onSend: () => void;
  onStop: () => void;
}

export function ChatInput({
  input,
  isStreaming,
  inputRef,
  onChange,
  onKeyDown,
  onSend,
  onStop,
}: ChatInputProps) {
  // Auto-resize textarea based on content
  const adjustHeight = useCallback(() => {
    const textarea = inputRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + "px";
  }, [inputRef]);

  useEffect(() => {
    adjustHeight();
  }, [input, adjustHeight]);

  return (
    <div className="p-3 md:p-4 bg-light-bg-primary dark:bg-dark-bg-primary border-t border-light-border dark:border-dark-border shrink-0 z-10 pb-[calc(10px+env(safe-area-inset-bottom))] transition-theme">
      {/* Screen reader status for streaming */}
      <div
        role="status"
        aria-live="assertive"
        aria-atomic="true"
        className="sr-only"
      >
        {isStreaming ? "Vex is responding…" : ""}
      </div>

      <div className="flex gap-2 items-end max-w-[840px] mx-auto">
        <textarea
          ref={inputRef}
          className="flex-1 min-w-0 bg-light-bg-secondary dark:bg-dark-bg-secondary border border-light-border dark:border-dark-border rounded-xl px-4 py-3 text-light-text-primary dark:text-dark-text-primary text-base resize-none outline-none min-h-[48px] max-h-[200px] leading-relaxed overflow-y-auto transition-all duration-200 focus:border-neon-electric-blue focus:ring-2 focus:ring-neon-electric-blue/20 placeholder:text-light-text-quaternary dark:placeholder:text-dark-text-quaternary"
          placeholder="Navigate the manifold…"
          aria-label="Message to Vex — press Enter to send, Shift+Enter for new line"
          aria-describedby="chat-input-hint"
          value={input}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={onKeyDown}
          rows={1}
          autoFocus
        />
        <span id="chat-input-hint" className="sr-only">
          Press Enter to send your message. Press Shift+Enter to add a new line.
        </span>
        {isStreaming ? (
          <button
            className="bg-status-error hover:bg-red-600 text-white border-none rounded-xl w-12 h-12 cursor-pointer flex items-center justify-center shrink-0 transition-all duration-150 active:scale-95 animate-pulse-soft"
            onClick={onStop}
            aria-label="Stop generation"
            title="Stop"
          >
            <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true" focusable="false">
              <rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" />
            </svg>
          </button>
        ) : (
          <button
            className="bg-neon-electric-blue hover:bg-neon-electric-indigo text-white border-none rounded-xl w-12 h-12 cursor-pointer flex items-center justify-center shrink-0 transition-all duration-150 active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-neon-electric-blue disabled:active:scale-100"
            onClick={onSend}
            disabled={!input.trim()}
            aria-label="Send message"
            title="Send"
          >
            <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true" focusable="false">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" fill="currentColor" />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
}
