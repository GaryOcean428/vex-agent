import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import { useNavigate } from "react-router-dom";
import "./CommandPalette.css";

/* ─── Types ─── */

interface Command {
  id: string;
  label: string;
  group: "Navigation" | "Actions" | "Chat";
  shortcut?: string[];
  action: () => void;
}

interface CommandPaletteProps {
  onNewChat?: () => void;
  onToggleHistory?: () => void;
}

/* ─── Highlight helper ─── */

function highlightMatch(text: string, query: string): ReactNode {
  if (!query) return text;

  const idx = text.toLowerCase().indexOf(query.toLowerCase());
  if (idx === -1) return text;

  const before = text.slice(0, idx);
  const match = text.slice(idx, idx + query.length);
  const after = text.slice(idx + query.length);

  return (
    <>
      {before}
      <mark className="cmd-highlight">{match}</mark>
      {after}
    </>
  );
}

/* ─── Search icon (inline SVG) ─── */

function SearchIcon() {
  return (
    <svg
      className="cmd-search-icon"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  );
}

/* ─── Group ordering ─── */

const GROUP_ORDER: Command["group"][] = ["Navigation", "Actions", "Chat"];

/* ─── Component ─── */

export function CommandPalette({
  onNewChat,
  onToggleHistory,
}: CommandPaletteProps) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  /* ── Build command list ── */

  const commands = useMemo<Command[]>(() => {
    const nav = (label: string, path: string): Command => ({
      id: `nav-${path}`,
      label,
      group: "Navigation",
      action: () => {
        navigate(path);
        setOpen(false);
      },
    });

    return [
      nav("Go to Chat", "/chat"),
      nav("Go to Dashboard", "/dashboard"),
      nav("Go to Consciousness", "/dashboard/consciousness"),
      nav("Go to Basins", "/dashboard/basins"),
      nav("Go to Graph", "/dashboard/graph"),
      nav("Go to Lifecycle", "/dashboard/lifecycle"),
      nav("Go to Cognition", "/dashboard/cognition"),
      nav("Go to Memory", "/dashboard/memory"),
      nav("Go to Telemetry", "/dashboard/telemetry"),
      nav("Go to Training", "/dashboard/training"),
      nav("Go to Governor", "/dashboard/governor"),
      nav("Go to Admin", "/dashboard/admin"),
      {
        id: "action-new-chat",
        label: "New Chat",
        group: "Actions",
        action: () => {
          onNewChat?.();
          setOpen(false);
        },
      },
      {
        id: "action-toggle-history",
        label: "Toggle History Panel",
        group: "Actions",
        action: () => {
          onToggleHistory?.();
          setOpen(false);
        },
      },
      {
        id: "chat-focus-input",
        label: "Focus Chat Input",
        group: "Chat",
        shortcut: ["Ctrl", "/"],
        action: () => {
          setOpen(false);
          // Defer so the palette closes before we try to focus
          requestAnimationFrame(() => {
            const textarea = document.querySelector<HTMLTextAreaElement>(
              ".chat-input",
            );
            textarea?.focus();
          });
        },
      },
    ];
  }, [navigate, onNewChat, onToggleHistory]);

  /* ── Filter commands ── */

  const filtered = useMemo(() => {
    if (!query.trim()) return commands;
    const q = query.toLowerCase();
    return commands.filter((cmd) => cmd.label.toLowerCase().includes(q));
  }, [commands, query]);

  /* ── Group filtered results ── */

  const grouped = useMemo(() => {
    const map = new Map<Command["group"], Command[]>();
    for (const cmd of filtered) {
      const list = map.get(cmd.group);
      if (list) {
        list.push(cmd);
      } else {
        map.set(cmd.group, [cmd]);
      }
    }
    // Return groups in canonical order, skipping empty ones
    return GROUP_ORDER.filter((g) => map.has(g)).map((g) => ({
      group: g,
      items: map.get(g)!,
    }));
  }, [filtered]);

  /* ── Flat list for keyboard navigation index ── */

  const flatItems = useMemo(
    () => grouped.flatMap((g) => g.items),
    [grouped],
  );

  /* ── Reset active index when filter changes ── */

  const handleQueryChange = useCallback((value: string) => {
    setQuery(value);
    setActiveIndex(0);
  }, []);

  /* ── Scroll active item into view ── */

  useEffect(() => {
    if (!listRef.current) return;
    const active = listRef.current.querySelector<HTMLElement>(
      '[data-active="true"]',
    );
    active?.scrollIntoView({ block: "nearest" });
  }, [activeIndex]);

  /* ── Open / close handlers ── */

  const handleOpen = useCallback(() => {
    setQuery("");
    setActiveIndex(0);
    setOpen(true);
    // Focus input on next frame after portal renders
    requestAnimationFrame(() => {
      inputRef.current?.focus();
    });
  }, []);

  const handleClose = useCallback(() => {
    setOpen(false);
  }, []);

  /* ── Execute the currently active command ── */

  const executeActive = useCallback(() => {
    const cmd = flatItems[activeIndex];
    cmd?.action();
  }, [flatItems, activeIndex]);

  /* ── Global keyboard shortcut: Ctrl+K / Cmd+K ── */

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        if (open) {
          handleClose();
        } else {
          handleOpen();
        }
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, handleOpen, handleClose]);

  /* ── Keyboard navigation inside the palette ── */

  const onInputKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          setActiveIndex((prev) =>
            prev < flatItems.length - 1 ? prev + 1 : 0,
          );
          break;
        case "ArrowUp":
          e.preventDefault();
          setActiveIndex((prev) =>
            prev > 0 ? prev - 1 : flatItems.length - 1,
          );
          break;
        case "Enter":
          e.preventDefault();
          executeActive();
          break;
        case "Escape":
          e.preventDefault();
          handleClose();
          break;
      }
    },
    [flatItems.length, executeActive, handleClose],
  );

  /* ── Render nothing if closed ── */

  if (!open) return null;

  /* ── Portal content ── */

  const palette = (
    <>
      {/* Backdrop */}
      <div
        className="cmd-backdrop"
        onClick={handleClose}
        aria-hidden="true"
      />

      {/* Dialog */}
      <div
        className="cmd-dialog"
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
      >
        {/* Search input */}
        <div className="cmd-input-wrapper">
          <SearchIcon />
          <input
            ref={inputRef}
            className="cmd-input"
            type="text"
            placeholder="Type a command..."
            aria-label="Search commands"
            aria-autocomplete="list"
            aria-controls="cmd-results"
            aria-activedescendant={
              flatItems[activeIndex]
                ? `cmd-item-${flatItems[activeIndex].id}`
                : undefined
            }
            value={query}
            onChange={(e) => handleQueryChange(e.target.value)}
            onKeyDown={onInputKeyDown}
          />
        </div>

        {/* Results */}
        <div
          className="cmd-results"
          id="cmd-results"
          ref={listRef}
          role="listbox"
          aria-label="Commands"
        >
          {flatItems.length === 0 ? (
            <div className="cmd-empty">No matching commands</div>
          ) : (
            grouped.map((section) => (
              <div key={section.group} role="group" aria-label={section.group}>
                <div className="cmd-group-label" aria-hidden="true">
                  {section.group}
                </div>
                {section.items.map((cmd) => {
                  const flatIdx = flatItems.indexOf(cmd);
                  const isActive = flatIdx === activeIndex;
                  return (
                    <div
                      key={cmd.id}
                      id={`cmd-item-${cmd.id}`}
                      className="cmd-item"
                      role="option"
                      aria-selected={isActive}
                      data-active={isActive}
                      onClick={() => cmd.action()}
                      onMouseEnter={() => setActiveIndex(flatIdx)}
                    >
                      <span className="cmd-item-label">
                        {highlightMatch(cmd.label, query)}
                      </span>
                      {cmd.shortcut && (
                        <span className="cmd-shortcut" aria-hidden="true">
                          {cmd.shortcut.map((key) => (
                            <kbd key={key} className="cmd-kbd">
                              {key}
                            </kbd>
                          ))}
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            ))
          )}
        </div>

        {/* Footer hints */}
        <div className="cmd-footer" aria-hidden="true">
          <span className="cmd-footer-hint">
            <kbd className="cmd-kbd">↑↓</kbd> navigate
          </span>
          <span className="cmd-footer-hint">
            <kbd className="cmd-kbd">↵</kbd> select
          </span>
          <span className="cmd-footer-hint">
            <kbd className="cmd-kbd">esc</kbd> close
          </span>
        </div>
      </div>
    </>
  );

  return createPortal(palette, document.body);
}
