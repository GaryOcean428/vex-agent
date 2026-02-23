import { useCallback, useEffect, type KeyboardEvent, type RefObject } from "react";
import "./ChatInput.css";

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
    <div className="chat-input-area">
      {/* Screen reader status for streaming */}
      <div
        role="status"
        aria-live="assertive"
        aria-atomic="true"
        className="sr-only"
      >
        {isStreaming ? "Vex is responding…" : ""}
      </div>

      <div className="input-wrapper">
        <textarea
          ref={inputRef}
          className="chat-input"
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
            className="stop-btn"
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
            className="send-btn"
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
