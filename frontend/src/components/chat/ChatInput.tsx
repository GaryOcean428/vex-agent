import type { KeyboardEvent, RefObject } from "react";
import "./ChatInput.css";

interface ChatInputProps {
  input: string;
  isStreaming: boolean;
  inputRef: RefObject<HTMLTextAreaElement | null>;
  onChange: (value: string) => void;
  onKeyDown: (e: KeyboardEvent<HTMLTextAreaElement>) => void;
  onSend: () => void;
}

export function ChatInput({
  input,
  isStreaming,
  inputRef,
  onChange,
  onKeyDown,
  onSend,
}: ChatInputProps) {
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
          disabled={isStreaming}
        />
        <span id="chat-input-hint" className="sr-only">
          Press Enter to send your message. Press Shift+Enter to add a new line.
        </span>
        <button
          className="send-btn"
          onClick={onSend}
          disabled={isStreaming || !input.trim()}
          aria-label="Send message"
          title="Send"
        >
          <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true" focusable="false">
            <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" fill="currentColor" />
          </svg>
        </button>
      </div>
    </div>
  );
}
