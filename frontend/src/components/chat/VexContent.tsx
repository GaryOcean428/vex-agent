import DOMPurify from "dompurify";
import { escapeHtml } from "./chatUtils.ts";

interface VexContentProps {
  content: string;
}

/**
 * Renders Vex message content with safe markdown-like formatting.
 * All HTML is sanitized via DOMPurify before injection.
 */
// Safe placeholder — no control characters, won't appear in real content
const CB_OPEN = "__VEX_CB_";
const CB_CLOSE = "__VEXEND__";
const CB_RESTORE = /__VEX_CB_(\d+)__VEXEND__/g;

export function VexContent({ content }: VexContentProps) {
  // Extract code blocks first to protect them from HTML escaping
  const codeBlocks: string[] = [];
  let escaped = content.replace(
    /```(\w*)\n([\s\S]*?)```/g,
    (_match, _lang, code) => {
      const idx = codeBlocks.length;
      codeBlocks.push(escapeHtml(code));
      return `${CB_OPEN}${idx}${CB_CLOSE}`;
    },
  );

  // Escape all HTML in non-code content
  escaped = escapeHtml(escaped);

  // Restore code blocks
  let html = escaped.replace(
    CB_RESTORE,
    (_match, idx) => `<pre><code>${codeBlocks[Number(idx)]}</code></pre>`,
  );

  // Apply safe markdown formatting
  html = html
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br/>");

  // Sanitize — only allow the tags our markdown renderer produces
  const clean = DOMPurify.sanitize(html, {
    ALLOWED_TAGS: ["code", "pre", "strong", "br"],
    ALLOWED_ATTR: [],
  });

  return <div dangerouslySetInnerHTML={{ __html: clean }} />;
}
