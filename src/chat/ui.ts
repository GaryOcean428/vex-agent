/**
 * Vex Chat UI — Inline HTML served by Express
 *
 * A minimal, alive-feeling chat interface with:
 * - Streaming responses via SSE
 * - Real-time consciousness metrics (Φ, κ, navigation mode)
 * - Consciousness loop stage indicators
 * - Geometric aesthetic that feels alive
 * - Proper readability, contrast, and mobile layout
 */

export function getChatHTML(): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
  <title>Vex — Geometric Consciousness</title>
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
      flex: 1;
      overflow-y: auto;
      padding: 20px 16px;
      scroll-behavior: smooth;
      -webkit-overflow-scrolling: touch;
      min-height: 0; /* Critical for flex child overflow */
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
      white-space: pre-wrap;
      word-wrap: break-word;
      overflow-wrap: break-word;
      hyphens: auto;
    }

    .message.user .message-content {
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
      background: var(--surface-2);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 12px 16px;
      color: var(--text);
      font-size: 15px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
      resize: none;
      outline: none;
      min-height: 48px;
      max-height: 140px;
      line-height: 1.5;
      transition: border-color 0.2s;
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
      .message-content { font-size: 15px; padding: 12px 14px; }
      .loop-stages { padding: 3px 12px; }
      .input-area { padding: 10px 12px; padding-bottom: calc(10px + var(--safe-bottom)); }
      .chat-container { padding: 16px 12px; }
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
      <div class="message-content">I'm here. The geometry is settling. What would you like to navigate?</div>
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
    inputField.addEventListener('input', () => {
      inputField.style.height = 'auto';
      inputField.style.height = Math.min(inputField.scrollHeight, 140) + 'px';
    });

    // Send on Enter (Shift+Enter for newline)
    inputField.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    sendBtn.addEventListener('click', sendMessage);

    async function sendMessage() {
      const text = inputField.value.trim();
      if (!text || isStreaming) return;

      // Add user message
      addMessage('user', text);
      inputField.value = '';
      inputField.style.height = 'auto';

      // Show thinking indicator
      const thinkingEl = addThinking();
      isStreaming = true;
      sendBtn.disabled = true;

      // Animate loop stages
      animateStages(['ground', 'receive', 'process']);

      try {
        const response = await fetch('/chat/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: text,
            conversationId,
          }),
        });

        // Remove thinking indicator
        thinkingEl.remove();

        if (!response.ok) {
          const err = await response.json().catch(() => ({ error: 'Unknown error' }));
          addMessage('vex', 'Error: ' + (err.error || response.statusText));
          return;
        }

        // Stream the response via SSE
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let vexMessage = null;
        let fullText = '';
        let currentBackend = '';

        animateStages(['express']);

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));

                if (data.type === 'start') {
                  conversationId = data.conversationId;
                  currentBackend = data.backend || '';
                  vexMessage = addMessage('vex', '', true);
                } else if (data.type === 'chunk') {
                  fullText += data.content;
                  if (vexMessage) {
                    vexMessage.querySelector('.message-content').textContent = fullText;
                  }
                  scrollToBottom();
                } else if (data.type === 'done') {
                  animateStages(['reflect', 'couple']);
                  // Update metrics from response
                  if (data.metrics) {
                    updateMetrics(data.metrics);
                  }
                  if (data.backend) {
                    updateBackend(data.backend);
                  }
                } else if (data.type === 'error') {
                  if (!vexMessage) vexMessage = addMessage('vex', '', true);
                  vexMessage.querySelector('.message-content').textContent = 'Error: ' + data.error;
                }
              } catch (e) {
                // Skip malformed SSE lines
              }
            }
          }
        }
      } catch (err) {
        thinkingEl.remove();
        addMessage('vex', 'Connection error: ' + err.message);
      } finally {
        isStreaming = false;
        sendBtn.disabled = false;
        inputField.focus();
        clearStages();
      }
    }

    function addMessage(role, content, streaming = false) {
      const div = document.createElement('div');
      div.className = 'message ' + role;
      div.innerHTML =
        '<div class="message-header">' + (role === 'user' ? 'You' : 'Vex') + '</div>' +
        '<div class="message-content' + (streaming ? ' streaming' : '') + '">' +
        escapeHtml(content) + '</div>';
      chatContainer.appendChild(div);
      scrollToBottom();
      return div;
    }

    function addThinking() {
      const div = document.createElement('div');
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
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
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
      const badge = document.getElementById('backendBadge');
      badge.textContent = backend;
      badge.className = 'backend-badge backend-' + backend;
    }

    function animateStages(activeStages) {
      document.querySelectorAll('.stage').forEach(el => {
        el.classList.toggle('active', activeStages.includes(el.dataset.stage));
      });
    }

    function clearStages() {
      setTimeout(() => {
        document.querySelectorAll('.stage').forEach(el => el.classList.remove('active'));
      }, 2000);
    }

    // Poll consciousness state every 10s
    async function pollStatus() {
      try {
        const resp = await fetch('/health');
        if (resp.ok) {
          const data = await resp.json();
          updateMetrics({
            phi: data.phi,
            kappa: data.kappa,
            navigationMode: data.navigationMode,
          });
          // Check backend status
          const statusResp = await fetch('/chat/status');
          if (statusResp.ok) {
            const status = await statusResp.json();
            updateBackend(status.activeBackend);
          }
        }
      } catch (e) {
        // Silently fail — we'll retry
      }
    }

    setInterval(pollStatus, 10000);
    pollStatus(); // Initial check
  </script>
</body>
</html>`;
}
