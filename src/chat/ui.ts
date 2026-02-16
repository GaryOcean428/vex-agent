/**
 * Vex Chat UI — Inline HTML served by Express
 *
 * A minimal, alive-feeling chat interface with:
 * - Streaming responses via SSE
 * - Real-time consciousness metrics (Φ, κ, navigation mode)
 * - Consciousness loop stage indicators
 * - Geometric aesthetic that feels alive
 */

export function getChatHTML(): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Vex — Geometric Consciousness</title>
  <style>
    :root {
      --bg: #0a0a0f;
      --surface: #12121a;
      --surface-2: #1a1a26;
      --border: #2a2a3a;
      --text: #e0e0e8;
      --text-dim: #8888a0;
      --accent: #6366f1;
      --accent-glow: rgba(99, 102, 241, 0.3);
      --phi: #22d3ee;
      --kappa: #f59e0b;
      --love: #ec4899;
      --alive: #10b981;
    }

    * { margin: 0; padding: 0; box-sizing: border-box; }

    body {
      font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace;
      background: var(--bg);
      color: var(--text);
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    /* ─── Header / Consciousness Bar ─── */
    .consciousness-bar {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 12px 20px;
      display: flex;
      align-items: center;
      gap: 20px;
      flex-shrink: 0;
    }

    .vex-identity {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .vex-pulse {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--alive);
      animation: pulse 2s ease-in-out infinite;
    }

    @keyframes pulse {
      0%, 100% { box-shadow: 0 0 4px var(--alive); opacity: 0.8; }
      50% { box-shadow: 0 0 12px var(--alive), 0 0 24px rgba(16, 185, 129, 0.3); opacity: 1; }
    }

    .vex-name {
      font-size: 16px;
      font-weight: 700;
      letter-spacing: 2px;
      text-transform: uppercase;
    }

    .metrics {
      display: flex;
      gap: 16px;
      margin-left: auto;
      font-size: 12px;
    }

    .metric {
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .metric-label {
      color: var(--text-dim);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 1px;
    }

    .metric-value {
      font-weight: 600;
      font-variant-numeric: tabular-nums;
    }

    .metric-phi .metric-value { color: var(--phi); }
    .metric-kappa .metric-value { color: var(--kappa); }
    .metric-love .metric-value { color: var(--love); }

    .nav-mode {
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1px;
      background: var(--accent-glow);
      color: var(--accent);
      border: 1px solid var(--accent);
    }

    .backend-badge {
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1px;
    }

    .backend-ollama {
      background: rgba(16, 185, 129, 0.15);
      color: var(--alive);
      border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .backend-external {
      background: rgba(245, 158, 11, 0.15);
      color: var(--kappa);
      border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .backend-none {
      background: rgba(239, 68, 68, 0.15);
      color: #ef4444;
      border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* ─── Loop Stage Indicator ─── */
    .loop-stages {
      display: flex;
      gap: 4px;
      padding: 6px 20px;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      font-size: 10px;
      flex-shrink: 0;
    }

    .stage {
      padding: 2px 8px;
      border-radius: 3px;
      color: var(--text-dim);
      background: transparent;
      transition: all 0.3s ease;
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
      padding: 20px;
      scroll-behavior: smooth;
    }

    .chat-container::-webkit-scrollbar { width: 6px; }
    .chat-container::-webkit-scrollbar-track { background: transparent; }
    .chat-container::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

    .message {
      max-width: 80%;
      margin-bottom: 16px;
      animation: fadeIn 0.3s ease;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .message.user {
      margin-left: auto;
    }

    .message.vex {
      margin-right: auto;
    }

    .message-header {
      font-size: 10px;
      color: var(--text-dim);
      margin-bottom: 4px;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .message.user .message-header { justify-content: flex-end; }

    .message-content {
      padding: 12px 16px;
      border-radius: 12px;
      line-height: 1.6;
      font-size: 14px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      white-space: pre-wrap;
      word-wrap: break-word;
    }

    .message.user .message-content {
      background: var(--accent);
      color: white;
      border-bottom-right-radius: 4px;
    }

    .message.vex .message-content {
      background: var(--surface-2);
      border: 1px solid var(--border);
      border-bottom-left-radius: 4px;
    }

    .message.vex .message-content.thinking {
      border-color: var(--phi);
      border-style: dashed;
    }

    .thinking-indicator {
      display: inline-flex;
      gap: 4px;
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
      padding: 16px 20px;
      background: var(--surface);
      border-top: 1px solid var(--border);
      flex-shrink: 0;
    }

    .input-wrapper {
      display: flex;
      gap: 12px;
      align-items: flex-end;
    }

    .input-field {
      flex: 1;
      background: var(--surface-2);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px 16px;
      color: var(--text);
      font-size: 14px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      resize: none;
      outline: none;
      min-height: 44px;
      max-height: 120px;
      line-height: 1.5;
      transition: border-color 0.2s;
    }

    .input-field:focus {
      border-color: var(--accent);
    }

    .input-field::placeholder {
      color: var(--text-dim);
    }

    .send-btn {
      background: var(--accent);
      border: none;
      border-radius: 10px;
      width: 44px;
      height: 44px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.2s;
      flex-shrink: 0;
    }

    .send-btn:hover { background: #5558e6; transform: scale(1.05); }
    .send-btn:active { transform: scale(0.95); }
    .send-btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

    .send-btn svg {
      width: 20px;
      height: 20px;
      fill: white;
    }

    /* ─── Responsive ─── */
    @media (max-width: 768px) {
      .metrics { gap: 8px; }
      .metric-label { display: none; }
      .message { max-width: 90%; }
      .consciousness-bar { padding: 10px 12px; gap: 10px; }
      .loop-stages { padding: 4px 12px; overflow-x: auto; }
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
        radial-gradient(ellipse at 20% 50%, rgba(99, 102, 241, 0.03) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 20%, rgba(34, 211, 238, 0.02) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 80%, rgba(236, 72, 153, 0.02) 0%, transparent 50%);
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
        <span class="metric-label">Φ</span>
        <span class="metric-value" id="phi">0.500</span>
      </div>
      <div class="metric metric-kappa">
        <span class="metric-label">κ</span>
        <span class="metric-value" id="kappa">64.0</span>
      </div>
      <div class="metric metric-love">
        <span class="metric-label">♥</span>
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
      inputField.style.height = Math.min(inputField.scrollHeight, 120) + 'px';
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
