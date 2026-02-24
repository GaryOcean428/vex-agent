import { useState, useCallback } from "react";
import DOMPurify from "dompurify";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import type { Components } from "react-markdown";

interface VexContentProps {
  content: string;
}

/**
 * Custom code component for react-markdown.
 * Code blocks (with a language-* className) get a <pre><code> wrapper
 * plus a Copy button. Inline code renders as plain <code>.
 */
function CodeBlock({
  className,
  children,
  node: _,
  ...htmlProps
}: React.ComponentPropsWithoutRef<"code"> & { node?: unknown }) {
  void _;
  const isBlock =
    typeof className === "string" && className.startsWith("language-");

  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    const text = String(children).replace(/\n$/, "");
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }, [children]);

  if (!isBlock) {
    return (
      <code className={className} {...htmlProps}>
        {children}
      </code>
    );
  }

  return (
    <div className="vex-codeblock">
      <button className="vex-copy-btn" onClick={handleCopy} type="button">
        {copied ? "Copied!" : "Copy"}
      </button>
      <pre>
        <code className={className} {...htmlProps}>
          {children}
        </code>
      </pre>
    </div>
  );
}

const markdownComponents: Components = {
  code: CodeBlock,
};

/**
 * Renders Vex message content with full markdown support.
 * - GFM: tables, strikethrough, task lists, autolinks
 * - Syntax-highlighted code blocks with copy button
 * - DOMPurify strips any raw HTML from the source before parsing
 */
export function VexContent({ content }: VexContentProps) {
  const sanitized = DOMPurify.sanitize(content, {
    ALLOWED_TAGS: [],
    ALLOWED_ATTR: [],
  });

  return (
    <div className="vex-markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={markdownComponents}
      >
        {sanitized}
      </ReactMarkdown>
    </div>
  );
}
