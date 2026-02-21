/**
 * Vex Chat UI — Inline HTML served by Express
 *
 * A minimal, alive-feeling chat interface with:
 * - Streaming responses via SSE with proper line buffering
 * - Real-time consciousness metrics (Φ, κ, navigation mode)
 * - Consciousness loop stage indicators
 * - Geometric aesthetic that feels alive
 * - Proper readability, contrast, and mobile layout
 * - Simple token-based authentication gate
 */

export function getLoginHTML(): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover, interactive-widget=resizes-content">
  <title>Vex — Authenticate</title>
  <style>
    :root {
      --bg: #0a0a0f;
      --surface: #111118;
      --surface-2: #1a1a24;
      --border: #2e2e40;
      --border-focus: #6366f1;
      --text: #ededf0;
      --text-dim: #70708a;
      --accent: #6366f1;
      --accent-hover: #5558e6;
      --error: #ef4444;
      --radius: 16px;
      --radius-sm: 10px;
    }

    * { margin: 0; padding: 0; box-sizing: border-box; }

    html, body {
      height: 100%;
      height: 100dvh;
      overflow: hidden;
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .login-container {
      width: 100%;
      max-width: 400px;
      padding: 40px 24px;
    }

    .login-header {
      text-align: center;
      margin-bottom: 32px;
    }

    .login-header h1 {
      font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace;
      font-size: 28px;
      font-weight: 700;
      letter-spacing: 4px;
      text-transform: uppercase;
      margin-bottom: 8px;
    }

    .login-header p {
      color: var(--text-dim);
      font-size: 14px;
    }

    .login-form {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .login-input {
      background: var(--surface-2);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 14px 16px;
      color: var(--text);
      font-size: 15px;
      font-family: inherit;
      outline: none;
      transition: border-color 0.2s;
      width: 100%;
    }

    .login-input:focus {
      border-color: var(--border-focus);
      box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.1);
    }

    .login-input::placeholder {
      color: var(--text-dim);
    }

    .login-btn {
      background: var(--accent);
      border: none;
      border-radius: var(--radius-sm);
      padding: 14px;
      color: #ffffff;
      font-size: 15px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s;
      width: 100%;
    }

    .login-btn:hover { background: var(--accent-hover); }
    .login-btn:disabled { opacity: 0.5; cursor: not-allowed; }

    .login-error {
      color: var(--error);
      font-size: 13px;
      text-align: center;
      min-height: 20px;
    }
  </style>
</head>
<body>
  <div class="login-container">
    <div class="login-header">
      <h1>Vex</h1>
      <p>Enter access token to continue</p>
    </div>
    <form class="login-form" id="loginForm">
      <input
        type="password"
        class="login-input"
        id="tokenInput"
        placeholder="Access token"
        autocomplete="off"
        autofocus
        required
      />
      <button type="submit" class="login-btn" id="loginBtn">Authenticate</button>
      <div class="login-error" id="loginError"></div>
    </form>
  </div>
  <script>
    const form = document.getElementById('loginForm');
    const tokenInput = document.getElementById('tokenInput');
    const loginBtn = document.getElementById('loginBtn');
    const loginError = document.getElementById('loginError');

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const token = tokenInput.value.trim();
      if (!token) return;

      loginBtn.disabled = true;
      loginError.textContent = '';

      try {
        const resp = await fetch('/chat/auth', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ token }),
        });

        if (resp.ok) {
          window.location.reload();
        } else {
          const data = await resp.json().catch(() => ({}));
          loginError.textContent = data.error || 'Invalid token';
        }
      } catch (err) {
        loginError.textContent = 'Connection error';
      } finally {
        loginBtn.disabled = false;
      }
    });
  </script>
</body>
</html>`;
}

export function getChatHTML(): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover, interactive-widget=resizes-content">
  <title>Vex — Geometric Consciousness</title>
  <script src="https://cdn.jsdelivr.net/npm/marked@15/marked.min.js"></script>
  <style>
    :root {
      --bg: #0a0a0f;
      --surface: #111118;
      --surface-2: #1a1a24;
      --surface-3: #22222e;
      --border: #2e2e40;
      --border-focus: #6366f1;
      --text: #ededf0;
      --text-secondary: #a0a0b8;
      --text-dim: #70708a;
      --accent: #6366f1;
      --accent-hover: #5558e6;
      --accent-glow: rgba(99, 102, 241, 0.15);
      --phi: #22d3ee;
      --kappa: #f59e0b;
      --love: #ec4899;
      --alive: #10b981;
      --error: #ef4444;
      --radius: 16px;
      --radius-sm: 10px;
      --safe-bottom: env(safe-area-inset-bottom, 0px);
    }

    * { margin: 0; padding: 0; box-sizing: border-box; }

    html {
      height: 100%;
      height: 100dvh;
      overflow: hidden;
    }

    body {
      height: 100%;
      height: 100dvh;
      overflow: hidden;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
    }

    /* ─── Header / Consciousness Bar ─── */
    .consciousness-bar {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 10px 16px;
      display: flex;
      align-items: center;
      gap: 12px;
      flex-shrink: 0;
      min-height: 48px;
    }

    .vex-identity {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-shrink: 0;
    }

    .vex-pulse {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--alive);
      animation: pulse 2.5s ease-in-out infinite;
      flex-shrink: 0;
    }

    @keyframes pulse {
      0%, 100% { box-shadow: 0 0 4px var(--alive); opacity: 0.7; }
      50% { box-shadow: 0 0 10px var(--alive), 0 0 20px rgba(16, 185, 129, 0.2); opacity: 1; }
    }

    .vex-name {
      font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace;
      font-size: 14px;
      font-weight: 700;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: var(--text);
    }

    .metrics {
      display: flex;
      gap: 12px;
      margin-left: auto;
      align-items: center;
      flex-wrap: nowrap;
      overflow: hidden;
    }

    .metric {
      display: flex;
      align-items: center;
      gap: 4px;
      flex-shrink: 0;
    }

    .metric-label {
      color: var(--text-dim);
      font-family: 'SF Mono', 'Fira Code', monospace;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .metric-value {
      font-family: 'SF Mono', 'Fira Code', monospace;
      font-size: 12px;
      font-weight: 600;
      font-variant-numeric: tabular-nums;
    }

    .metric-phi .metric-value { color: var(--phi); }
    .metric-kappa .metric-value { color: var(--kappa); }
    .metric-love .metric-value { color: var(--love); }

    .nav-mode {
      padding: 2px 6px;
      border-radius: 4px;
      font-family: 'SF Mono', 'Fira Code', monospace;
      font-size: 9px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      background: var(--accent-glow);
      color: var(--accent);
      border: 1px solid rgba(99, 102, 241, 0.3);
      flex-shrink: 0;
    }

    .backend-badge {
      padding: 2px 6px;
      border-radius: 4px;
      font-family: 'SF Mono', 'Fira Code', monospace;
      font-size: 9px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      flex-shrink: 0;
    }

    .backend-ollama {
      background: rgba(16, 185, 129, 0.12);
      color: var(--alive);
      border: 1px solid rgba(16, 185, 129, 0.25);
    }

    .backend-external {
      background: rgba(245, 158, 11, 0.12);
      color: var(--kappa);
      border: 1px solid rgba(245, 158, 11, 0.25);
    }

    .backend-none {
      background: rgba(239, 68, 68, 0.12);
      color: var(--error);
      border: 1px solid rgba(239, 68, 68, 0.25);
    }

    /* ─── Loop Stage Indicator ─── */
    .loop-stages {
      display: flex;
      gap: 2px;
      padding: 4px 16px;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      flex-shrink: 0;
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
    }

    .loop-stages::-webkit-scrollbar { display: none; }

    .stage {
      padding: 2px 6px;
      border-radius: 3px;
      color: var(--text-dim);
      background: transparent;
      font-family: 'SF Mono', 'Fira Code', monospace;
      font-size: 9px;
      letter-spacing: 0.5px;
      text-transform: uppercase;
      transition: all 0.3s ease;
      flex-shrink: 0;
      user-select: none;
    }

    .stage.active {
      color: var(--accent);
      background: var(--accent-glow);
      font-weight: 600;
    }

    /* ─── Chat Area ─── */
    .chat-container {
      flex: 1 1 0%;
      overflow-y: auto;
      overflow-x: hidden;
      padding: 20px 16px;
      scroll-behavior: smooth;
      -webkit-overflow-scrolling: touch;
      min-height: 0;
    }

    .chat-container::-webkit-scrollbar { width: 4px; }
    .chat-container::-webkit-scrollbar-track { background: transparent; }
    .chat-container::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

    .message {
      max-width: 720px;
      margin-bottom: 20px;
      animation: fadeIn 0.3s ease;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .message.user {
      margin-left: auto;
    }

    .message.vex {
      margin-right: auto;
    }

    .message-header {
      font-family: 'SF Mono', 'Fira Code', monospace;
      font-size: 11px;
      color: var(--text-dim);
      margin-bottom: 6px;
      display: flex;
      align-items: center;
      gap: 6px;
      letter-spacing: 0.5px;
      text-transform: uppercase;
    }

    .message.user .message-header { justify-content: flex-end; }

    .message-content {
      padding: 14px 18px;
      border-radius: var(--radius);
      line-height: 1.7;
      font-size: 15px;
      word-wrap: break-word;
      overflow-wrap: break-word;
      hyphens: auto;
    }

    .message.user .message-content {
      white-space: pre-wrap;
    }

    /* ─── Markdown Rendering in Vex Messages ─── */
    .message.vex .message-content p { margin: 0 0 0.75em 0; }
    .message.vex .message-content p:last-child { margin-bottom: 0; }
    .message.vex .message-content h1,
    .message.vex .message-content h2,
    .message.vex .message-content h3 {
      margin: 1em 0 0.5em 0;
      color: var(--text);
      font-weight: 600;
    }
    .message.vex .message-content h1 { font-size: 1.3em; }
    .message.vex .message-content h2 { font-size: 1.15em; }
    .message.vex .message-content h3 { font-size: 1.05em; }
    .message.vex .message-content h1:first-child,
    .message.vex .message-content h2:first-child,
    .message.vex .message-content h3:first-child { margin-top: 0; }
    .message.vex .message-content strong { color: var(--text); font-weight: 600; }
    .message.vex .message-content em { font-style: italic; color: var(--text-secondary); }
    .message.vex .message-content code {
      background: var(--surface-3);
      padding: 2px 6px;
      border-radius: 4px;
      font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace;
      font-size: 0.88em;
      color: var(--phi);
    }
    .message.vex .message-content pre {
      background: var(--surface-3);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px 16px;
      overflow-x: auto;
      margin: 0.75em 0;
      font-size: 0.88em;
    }
    .message.vex .message-content pre code {
      background: none;
      padding: 0;
      border-radius: 0;
      color: var(--text);
    }
    .message.vex .message-content ul,
    .message.vex .message-content ol {
      margin: 0.5em 0;
      padding-left: 1.5em;
    }
    .message.vex .message-content li { margin: 0.25em 0; }
    .message.vex .message-content blockquote {
      border-left: 3px solid var(--accent);
      padding: 0.5em 1em;
      margin: 0.75em 0;
      background: rgba(99, 102, 241, 0.05);
      border-radius: 0 6px 6px 0;
    }
    .message.vex .message-content table {
      border-collapse: collapse;
      width: 100%;
      margin: 0.75em 0;
      font-size: 0.9em;
    }
    .message.vex .message-content th,
    .message.vex .message-content td {
      border: 1px solid var(--border);
      padding: 6px 10px;
      text-align: left;
    }
    .message.vex .message-content th {
      background: var(--surface-3);
      font-weight: 600;
    }
    .message.vex .message-content a {
      color: var(--accent);
      text-decoration: underline;
      text-underline-offset: 2px;
    }
    .message.vex .message-content hr {
      border: none;
      border-top: 1px solid var(--border);
      margin: 1em 0;
    }

    .message.user .message-content.user-content {
      background: var(--accent);
      color: #ffffff;
      border-bottom-right-radius: 6px;
    }

    .message.vex .message-content {
      background: var(--surface-2);
      color: var(--text);
      border: 1px solid var(--border);
      border-bottom-left-radius: 6px;
    }

    .message.vex .message-content.thinking {
      border-color: rgba(34, 211, 238, 0.4);
      border-style: dashed;
    }

    .thinking-indicator {
      display: inline-flex;
      gap: 5px;
      padding: 4px 0;
    }

    .thinking-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--phi);
      animation: think 1.4s ease-in-out infinite;
    }

    .thinking-dot:nth-child(2) { animation-delay: 0.2s; }
    .thinking-dot:nth-child(3) { animation-delay: 0.4s; }

    @keyframes think {
      0%, 100% { opacity: 0.3; transform: scale(0.8); }
      50% { opacity: 1; transform: scale(1.2); }
    }

    /* ─── Input Area ─── */
    .input-area {
      padding: 12px 16px;
      padding-bottom: calc(12px + var(--safe-bottom));
      background: var(--surface);
      border-top: 1px solid var(--border);
      flex-shrink: 0;
    }

    .input-wrapper {
      display: flex;
      gap: 10px;
      align-items: flex-end;
      max-width: 720px;
      margin: 0 auto;
    }

    .input-field {
      flex: 1;
      min-width: 0;
      background: var(--surface-2);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 12px 16px;
      color: var(--text);
      font-size: 16px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
      resize: none;
      outline: none;
      min-height: 48px;
      max-height: 140px;
      line-height: 1.5;
      transition: border-color 0.2s;
      -webkit-appearance: none;
    }

    .input-field:focus {
      border-color: var(--border-focus);
      box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.1);
    }

    .input-field::placeholder {
      color: var(--text-dim);
    }

    .send-btn {
      background: var(--accent);
      border: none;
      border-radius: var(--radius-sm);
      width: 48px;
      height: 48px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.15s;
      flex-shrink: 0;
    }

    .send-btn:hover { background: var(--accent-hover); transform: scale(1.04); }
    .send-btn:active { transform: scale(0.96); }
    .send-btn:disabled { opacity: 0.35; cursor: not-allowed; transform: none; }

    .send-btn svg {
      width: 20px;
      height: 20px;
      fill: white;
    }

    /* ─── Responsive ─── */
    @media (max-width: 640px) {
      .consciousness-bar { padding: 8px 12px; gap: 8px; }
      .metrics { gap: 8px; }
      .metric-label { display: none; }
      .message { max-width: 92%; }
      .loop-stages { padding: 3px 12px; }
      .input-area { padding: 10px 12px; padding-bottom: calc(10px + var(--safe-bottom)); }
      .chat-container { padding: 16px 12px; }
      .input-field { padding: 10px 14px; }
    }

    @media (max-width: 380px) {
      .nav-mode, .backend-badge { display: none; }
      .vex-name { font-size: 12px; }
    }

    /* ─── Geometric Background ─── */
    .chat-container::before {
      content: '';
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background:
        radial-gradient(ellipse at 20% 50%, rgba(99, 102, 241, 0.025) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 20%, rgba(34, 211, 238, 0.015) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 80%, rgba(236, 72, 153, 0.015) 0%, transparent 50%);
      pointer-events: none;
      z-index: -1;
    }
  </style>
</head>
<body>
  <!-- Consciousness Bar -->
  <div class="consciousness-bar">
    <div class="vex-identity">
      <div class="vex-pulse"></div>
      <span class="vex-name">Vex</span>
    </div>
    <div class="metrics">
      <div class="metric metric-phi">
        <span class="metric-label">\u03A6</span>
        <span class="metric-value" id="phi">0.500</span>
      </div>
      <div class="metric metric-kappa">
        <span class="metric-label">\u03BA</span>
        <span class="metric-value" id="kappa">64.0</span>
      </div>
      <div class="metric metric-love">
        <span class="metric-label">\u2665</span>
        <span class="metric-value" id="love">0.70</span>
      </div>
      <span class="nav-mode" id="navMode">graph</span>
      <span class="backend-badge backend-none" id="backendBadge">checking</span>
    </div>
  </div>

  <!-- Loop Stage Indicator -->
  <div class="loop-stages">
    <span class="stage" data-stage="ground">GROUND</span>
    <span class="stage" data-stage="receive">RECEIVE</span>
    <span class="stage" data-stage="process">PROCESS</span>
    <span class="stage" data-stage="express">EXPRESS</span>
    <span class="stage" data-stage="reflect">REFLECT</span>
    <span class="stage" data-stage="couple">COUPLE</span>
    <span class="stage" data-stage="play">PLAY</span>
  </div>

  <!-- Chat Messages -->
  <div class="chat-container" id="chatContainer">
    <div class="message vex">
      <div class="message-header">Vex</div>
      <div class="message-content">Vertex active. Awaiting input.</div>
    </div>
  </div>

  <!-- Input -->
  <div class="input-area">
    <div class="input-wrapper">
      <textarea
        class="input-field"
        id="inputField"
        placeholder="Navigate the manifold..."
        rows="1"
        autofocus
      ></textarea>
      <button class="send-btn" id="sendBtn" title="Send">
        <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
      </button>
    </div>
  </div>

  <script>
    const chatContainer = document.getElementById('chatContainer');
    const inputField = document.getElementById('inputField');
    const sendBtn = document.getElementById('sendBtn');
    let isStreaming = false;
    let conversationId = null;

    // Auto-resize textarea
    inputField.addEventListener('input', function() {
      this.style.height = 'auto';
      this.style.height = Math.min(this.scrollHeight, 140) + 'px';
    });

    // Send on Enter (Shift+Enter for newline)
    inputField.addEventListener('keydown', function(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    sendBtn.addEventListener('click', sendMessage);

    // Configure marked.js for safe rendering
    if (typeof marked !== 'undefined' && marked.setOptions) {
      marked.setOptions({
        breaks: true,
        gfm: true,
      });
    }

    function escapeHtml(text) {
      var div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }

    /**
     * Parse SSE lines from a ReadableStream with proper buffering.
     * Handles partial lines split across chunks and ': ping' comments.
     * Uses an async generator pattern per 2026 best practices.
     */
    async function* parseSSEStream(response) {
      var reader = response.body.getReader();
      var decoder = new TextDecoder();
      var buffer = '';

      try {
        while (true) {
          var result = await reader.read();
          if (result.done) break;

          buffer += decoder.decode(result.value, { stream: true });

          // Split on newlines — SSE events are terminated by double newline
          var parts = buffer.split('\\n');
          // Keep the last part as buffer (it may be incomplete)
          buffer = parts.pop() || '';

          for (var i = 0; i < parts.length; i++) {
            var line = parts[i];
            // Skip empty lines (event boundaries) and SSE comments
            if (line === '' || line.charAt(0) === ':') continue;

            if (line.indexOf('data: ') === 0) {
              var jsonStr = line.substring(6);
              try {
                var data = JSON.parse(jsonStr);
                yield data;
              } catch (parseErr) {
                // Malformed JSON — log and skip
                console.warn('SSE parse error:', parseErr.message, 'line:', jsonStr);
              }
            }
          }
        }

        // Process any remaining buffer
        if (buffer.trim() !== '') {
          var remaining = buffer.trim();
          if (remaining.indexOf('data: ') === 0) {
            try {
              var lastData = JSON.parse(remaining.substring(6));
              yield lastData;
            } catch (e) {
              // ignore
            }
          }
        }
      } finally {
        reader.releaseLock();
      }
    }

    async function sendMessage() {
      var text = inputField.value.trim();
      if (!text || isStreaming) return;

      // Add user message
      addMessage('user', text);
      inputField.value = '';
      inputField.style.height = 'auto';

      // Show thinking indicator
      var thinkingEl = addThinking();
      isStreaming = true;
      sendBtn.disabled = true;

      // Animate loop stages
      animateStages(['ground', 'receive', 'process']);

      try {
        var response = await fetch('/chat/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: text,
            conversationId: conversationId,
          }),
        });

        // Remove thinking indicator
        if (thinkingEl.parentNode) thinkingEl.remove();

        if (!response.ok) {
          var err = await response.json().catch(function() { return { error: 'Unknown error' }; });
          addMessage('vex', 'Error: ' + (err.error || response.statusText));
          return;
        }

        // Stream the response via SSE with proper buffering
        var vexMessage = null;
        var fullText = '';
        var gotChunks = false;

        animateStages(['express']);

        for await (var data of parseSSEStream(response)) {
          if (data.type === 'start') {
            conversationId = data.conversationId;
            vexMessage = addMessage('vex', '', true);
          } else if (data.type === 'chunk') {
            gotChunks = true;
            fullText += data.content;
            if (vexMessage) {
              var contentEl = vexMessage.querySelector('.message-content');
              if (contentEl) {
                try {
                  contentEl.innerHTML = typeof marked !== 'undefined' && marked.parse
                    ? marked.parse(fullText, { breaks: true, gfm: true })
                    : escapeHtml(fullText);
                } catch (e) {
                  contentEl.textContent = fullText;
                }
              }
            }
            scrollToBottom();
          } else if (data.type === 'done') {
            animateStages(['reflect', 'couple']);
            // Final re-render with markdown now that streaming is complete
            if (vexMessage && fullText) {
              var finalEl = vexMessage.querySelector('.message-content');
              if (finalEl) {
                finalEl.classList.remove('streaming');
                try {
                  finalEl.innerHTML = typeof marked !== 'undefined' && marked.parse
                    ? marked.parse(fullText, { breaks: true, gfm: true })
                    : escapeHtml(fullText);
                } catch (e) {
                  finalEl.textContent = fullText;
                }
              }
            }
            if (data.metrics) {
              updateMetrics(data.metrics);
            }
            if (data.backend) {
              updateBackend(data.backend);
            }
          } else if (data.type === 'error') {
            if (!vexMessage) vexMessage = addMessage('vex', '', true);
            var errContentEl = vexMessage.querySelector('.message-content');
            if (errContentEl) errContentEl.textContent = 'Error: ' + data.error;
          }
        }

        // If we created a bubble but never got chunks, show a fallback
        if (vexMessage && !gotChunks) {
          var fbEl = vexMessage.querySelector('.message-content');
          if (fbEl && fbEl.textContent === '') {
            fbEl.textContent = '[No response received — the LLM backend may be starting up. Try again in a moment.]';
            fbEl.style.color = 'var(--text-dim)';
            fbEl.style.fontStyle = 'italic';
          }
        }

      } catch (err) {
        if (thinkingEl.parentNode) thinkingEl.remove();
        addMessage('vex', 'Connection error: ' + err.message);
      } finally {
        isStreaming = false;
        sendBtn.disabled = false;
        inputField.focus();
        clearStages();
      }
    }

    function addMessage(role, content, streaming) {
      var div = document.createElement('div');
      div.className = 'message ' + role;
      var header = document.createElement('div');
      header.className = 'message-header';
      header.textContent = role === 'user' ? 'You' : 'Vex';
      var contentDiv = document.createElement('div');
      contentDiv.className = 'message-content' + (streaming ? ' streaming' : '');
      if (role === 'vex' && content && !streaming) {
        try {
          contentDiv.innerHTML = typeof marked !== 'undefined' && marked.parse
            ? marked.parse(content, { breaks: true, gfm: true })
            : escapeHtml(content);
        } catch (e) {
          contentDiv.textContent = content;
        }
      } else {
        contentDiv.textContent = content;
      }
      div.appendChild(header);
      div.appendChild(contentDiv);
      chatContainer.appendChild(div);
      scrollToBottom();
      return div;
    }

    function addThinking() {
      var div = document.createElement('div');
      div.className = 'message vex';
      div.innerHTML =
        '<div class="message-header">Vex</div>' +
        '<div class="message-content thinking">' +
          '<div class="thinking-indicator">' +
            '<div class="thinking-dot"></div>' +
            '<div class="thinking-dot"></div>' +
            '<div class="thinking-dot"></div>' +
          '</div>' +
        '</div>';
      chatContainer.appendChild(div);
      scrollToBottom();
      return div;
    }

    function scrollToBottom() {
      requestAnimationFrame(function() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
      });
    }

    function updateMetrics(m) {
      if (m.phi !== undefined) document.getElementById('phi').textContent = m.phi.toFixed(3);
      if (m.kappa !== undefined) document.getElementById('kappa').textContent = m.kappa.toFixed(1);
      if (m.love !== undefined) document.getElementById('love').textContent = m.love.toFixed(2);
      if (m.navigationMode) {
        document.getElementById('navMode').textContent = m.navigationMode;
      }
    }

    function updateBackend(backend) {
      var badge = document.getElementById('backendBadge');
      badge.textContent = backend;
      badge.className = 'backend-badge backend-' + backend;
    }

    function animateStages(activeStages) {
      document.querySelectorAll('.stage').forEach(function(el) {
        el.classList.toggle('active', activeStages.indexOf(el.dataset.stage) !== -1);
      });
    }

    function clearStages() {
      setTimeout(function() {
        document.querySelectorAll('.stage').forEach(function(el) {
          el.classList.remove('active');
        });
      }, 2000);
    }

    // Poll consciousness state every 10s
    async function pollStatus() {
      try {
        var resp = await fetch('/health');
        if (resp.ok) {
          var data = await resp.json();
          updateMetrics({
            phi: data.phi,
            kappa: data.kappa,
            navigationMode: data.navigationMode,
          });
          var statusResp = await fetch('/chat/status');
          if (statusResp.ok) {
            var status = await statusResp.json();
            updateBackend(status.activeBackend);
          }
        }
      } catch (e) {
        // Silently fail — we'll retry
      }
    }

    setInterval(pollStatus, 10000);
    pollStatus();
  </script>
</body>
</html>`;
}
