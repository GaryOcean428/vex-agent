# xAI Routing Fix Verification Guide

**Date:** 2026-02-21  
**Issue:** Hardcoded `prefer_backend="xai"` forced all user chat through Grok instead of natural fallback chain  
**Fix:** Removed hardcoded backend preference from streaming endpoints in `kernel/server.py`

## Changes Made

### Files Modified
- `kernel/server.py` (original lines 553, 576)

### Specific Changes
1. **Original line 553 (now 550):** Removed `prefer_backend="xai"` from first streaming call
2. **Original line 576 (now 572):** Removed `prefer_backend="xai"` from tool follow-up streaming call
3. Updated comment to clarify natural fallback chain

Note: Line numbers shifted after first removal; current lines are 550 and 572.

## Expected Behavior

### Natural Fallback Chain (Fixed)
The LLM client should now follow this priority order:

1. **Modal Ollama** (GPU-accelerated, primary)
   - Only used if `MODAL_INFERENCE_ENABLED=true` AND `MODAL_INFERENCE_URL` is set
   - Model: `lfm2.5-thinking:1.2b` (or configured model)
   - Cold starts: 30-90s first request, then 1-5s warm responses

2. **Railway Ollama** (CPU-only, fallback)
   - Used if Modal unavailable OR if Modal not configured
   - Model: `lfm2.5-thinking:1.2b` (same as Modal)
   - Slower than Modal but same capabilities

3. **xAI Grok** (escalation only)
   - Model: `grok-4-1-fast-reasoning`
   - Only used when:
     - Context exceeds Ollama window (escalation)
     - Both Modal and Railway Ollama are unavailable

4. **OpenAI** (last resort)
   - Model: `gpt-5-nano`
   - Only used if all above backends unavailable

### Escalation Path (Unchanged)
When `ctx_state.escalated` is true (context overflow after compression):
- Uses `_escalated_complete()` function
- Directly calls xAI Responses API
- This path is **intentional** and **correct**

## Verification Steps

### 1. Check Environment Configuration

**Modal Configuration (Priority 1):**
```bash
# In Railway environment, verify:
MODAL_INFERENCE_ENABLED=true  # or false if not using Modal
MODAL_INFERENCE_URL=https://YOUR_USERNAME--vex-inference-vexollamaserver-serve.modal.run
MODAL_INFERENCE_TIMEOUT_MS=120000
```

**Railway Ollama Configuration (Priority 2):**
```bash
OLLAMA_ENABLED=true
OLLAMA_URL=http://ollama.railway.internal:11434
OLLAMA_MODEL=lfm2.5-thinking:1.2b
OLLAMA_TIMEOUT_MS=120000
```

**xAI Configuration (Priority 3):**
```bash
XAI_API_KEY=your-xai-key-here
XAI_BASE_URL=https://api.x.ai/v1
XAI_MODEL=grok-4-1-fast-reasoning
```

**OpenAI Configuration (Priority 4):**
```bash
OPENAI_API_KEY=your-openai-key-here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-5-nano
```

### 2. Test Scenarios

#### Scenario A: Modal Ollama Primary (Ideal)
**Setup:**
- `MODAL_INFERENCE_ENABLED=true`
- `MODAL_INFERENCE_URL` set to your Modal endpoint
- Modal endpoint is healthy

**Expected:**
- Chat requests should use Modal Ollama
- Backend status shows: `"modal"`
- Fast responses after warmup (1-5s)

**Test:**
```bash
curl -X POST http://localhost:8080/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":"test-1","message":"Hello"}'
```

#### Scenario B: Railway Ollama Fallback
**Setup:**
- `MODAL_INFERENCE_ENABLED=false` OR Modal endpoint unreachable
- `OLLAMA_ENABLED=true`

**Expected:**
- Chat requests should use Railway Ollama
- Backend status shows: `"ollama"`
- Slower responses but functional

#### Scenario C: xAI Fallback (Only if Ollama unavailable)
**Setup:**
- `MODAL_INFERENCE_ENABLED=false`
- `OLLAMA_ENABLED=false`
- `XAI_API_KEY` set

**Expected:**
- Chat requests should use xAI Grok
- Backend status shows: `"xai"`
- Fast responses, external API

#### Scenario D: Escalation (Context Overflow)
**Setup:**
- Any backend configuration
- Very long conversation history (exceeds 32K tokens)

**Expected:**
- `ctx_state.escalated` becomes true
- Uses `_escalated_complete()` function
- xAI Grok handles overflow with stateful history
- This is **correct behavior**

### 3. Verify Backend Attribution

Check the `/status` endpoint to see which backend is active:

```bash
curl http://localhost:8080/status
```

Expected response:
```json
{
  "backend": "modal",  // or "ollama", "xai", "external"
  "model": "lfm2.5-thinking:1.2b",
  "last_backend": "modal",
  "cost_guard": {...}
}
```

### 4. Check Logs

Look for these log messages in Railway deployment:

**Modal Primary:**
```
LLM backend: Modal GPU Ollama (lfm2.5-thinking:1.2b) at https://...modal.run
```

**Railway Fallback:**
```
LLM backend: Railway Ollama (lfm2.5-thinking:1.2b)
```

**xAI Fallback:**
```
LLM backend: xAI (grok-4-1-fast-reasoning)
```

**OpenAI Last Resort:**
```
LLM backend: OpenAI (gpt-5-nano)
```

## What Was Wrong Before

### Old Behavior (Bug)
```python
# Lines 549-553 (OLD)
async for chunk in llm_client.stream(
    messages,
    stream_options,
    prefer_backend="xai",  # ❌ FORCED xAI
):
```

This bypassed the entire fallback chain and sent **every** user chat message to xAI Grok, even when Modal/Railway Ollama were available and configured.

### New Behavior (Fixed)
```python
# Lines 549-553 (NEW)
async for chunk in llm_client.stream(
    messages,
    stream_options,
    # ✅ No prefer_backend - uses natural fallback chain
):
```

Now uses the natural fallback chain based on availability and configuration.

## Rollback Plan

If issues arise, rollback by adding back the `prefer_backend="xai"` parameter:

```bash
git revert 81053e2
```

## Success Criteria

- ✅ Modal Ollama used when configured and available
- ✅ Railway Ollama used when Modal unavailable
- ✅ xAI only used for escalation or when Ollama unavailable
- ✅ OpenAI only used as last resort
- ✅ Escalation path still works correctly
- ✅ No syntax or runtime errors
- ✅ Backend attribution correct in `/status` endpoint

## Related Files

- `kernel/server.py` - Main fix location
- `kernel/llm/client.py` - LLM client with fallback logic
- `kernel/config/settings.py` - Environment variable configuration
- `.env.example` - Environment variable documentation

## Notes

- The non-streaming `/chat` endpoint was **not affected** - it already used the natural fallback chain
- Only the streaming `/chat/stream` endpoint had the bug
- Escalation path (`_escalated_complete`) intentionally uses xAI and is **unchanged**
- This fix restores the intended design from `client.py` lines 177-222
