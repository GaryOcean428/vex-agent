import { useCallback, useEffect, useState } from "react";
import { API } from "../../config/api-routes.ts";
import "./ChatHistory.css";

interface ConversationSummary {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  message_count: number;
  preview: string;
}

interface ChatHistoryProps {
  collapsed: boolean;
  activeConversationId: string | null;
  onSelect: (id: string) => void;
  onNewChat: () => void;
}

export function ChatHistory({ open, activeConversationId, onSelect, onNewChat }: ChatHistoryProps) {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [loading, setLoading] = useState(false);

  const loadConversations = useCallback(async () => {
    setLoading(true);
    try {
      const resp = await fetch(API.conversations);
      if (resp.ok) {
        const data = await resp.json();
        setConversations(data.conversations ?? data ?? []);
      }
    } catch {
      // silent
    } finally {
      setLoading(false);
    }
  }, []);

  // Load on mount and when panel opens
  useEffect(() => {
    loadConversations();
  }, [loadConversations]);

  // Refresh list when active conversation changes (new message sent)
  useEffect(() => {
    if (open && activeConversationId) {
      // Debounce: wait 500ms for the server to persist the message
      const timer = setTimeout(loadConversations, 500);
      return () => clearTimeout(timer);
    }
  }, [open, activeConversationId, loadConversations]);

  return (
    <aside
      className={`chat-history-panel ${open ? "open" : "collapsed"}`}
      aria-label="Chat history"
      aria-hidden={!open}
    >
      {open && (
        <>
          <div className="chat-history-header">
            <span className="chat-history-title">Conversations</span>
            <button
              className="chat-history-new"
              onClick={onNewChat}
              aria-label="New conversation"
              title="New conversation"
            >
              +
            </button>
          </div>

          {loading && conversations.length === 0 && (
            <div className="chat-history-loading">Loading...</div>
          )}

          <div className="chat-history-list">
            {conversations.length === 0 && !loading && (
              <div className="chat-history-empty">No past conversations</div>
            )}
            {conversations.map((conv) => (
              <button
                key={conv.id}
                className={`chat-history-item ${conv.id === activeConversationId ? "active" : ""}`}
                onClick={() => onSelect(conv.id)}
              >
                <span className="chat-history-item-title">
                  {conv.title || "Untitled"}
                </span>
                <span className="chat-history-item-meta">
                  {conv.message_count} msgs &middot; {formatRelativeTime(conv.updated_at)}
                </span>
                {conv.preview && (
                  <span className="chat-history-item-preview">{conv.preview}</span>
                )}
              </button>
            ))}
          </div>
        </>
      )}
    </aside>
  );
}

function formatRelativeTime(epoch: number): string {
  const now = Date.now() / 1000;
  const diff = now - epoch;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  if (diff < 604800) return `${Math.floor(diff / 86400)}d ago`;
  return new Date(epoch * 1000).toLocaleDateString();
}
