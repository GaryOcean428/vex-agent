# Fine-tuning open-weight models for Ollama in 2026

OpenAI's **gpt-oss** family — released August 2025 under Apache 2.0 — is the company's first open-weight LLM since GPT-2, and it uses a proprietary structured format called **Harmony** that you'll need to understand before fine-tuning. However, for your specific setup (A10G 24GB, Ollama on Modal, QIG reasoning), **Qwen3-14B or Qwen3-30B-A3B is likely the better practical choice** over gpt-oss, due to superior fine-tuning tooling, native thinking mode, and a simpler chat template. This report covers the full landscape: what OpenAI released, how Harmony works, and a head-to-head comparison of every viable model for your pipeline.

---

## OpenAI's gpt-oss models broke a six-year open-weight drought

On August 5, 2025, OpenAI released **gpt-oss-120b** (116.8B total, **5.1B active** per token, 128 MoE experts, top-4 routing) and **gpt-oss-20b** (20.9B total, **3.6B active**, 32 MoE experts). Both are autoregressive MoE Transformers using RMSNorm, GQA with 8 KV heads, SwiGLU activations, RoPE, and a **128K context** window extended via YaRN. The architecture descends from GPT-2/GPT-3 but incorporates modern innovations — alternating banded-window and dense attention, learned attention-sink biases, and MXFP4 quantization of MoE weights.

The models ship under **Apache 2.0** — more permissive than Meta's Llama license — and are available on HuggingFace (`openai/gpt-oss-120b` with 4.6M+ downloads; `openai/gpt-oss-20b` with 7.5M+). A follow-up release in October 2025, **gpt-oss-safeguard**, provides safety-classifier variants fine-tuned for Trust & Safety workflows. No additional open-weight models have been released by OpenAI in 2026 as of March.

For your A10G setup, **gpt-oss-20b is the relevant model** — its 3.6B active parameters and ~21B total make it architecturally comparable to your current GLM-4-9B-Flash. The 120b variant requires an 80GB GPU (H100 or MI300X) with MXFP4 quantization, putting it out of reach for single-A10G fine-tuning. Community fine-tuning activity is strong: HuggingFace already lists **92+ fine-tunes and 26 LoRA adapters** for gpt-oss-120b alone.

---

## Harmony format is OpenAI's next-generation structured protocol

The "harmony format" you asked about is real and thoroughly documented. It is OpenAI's **structured token protocol** introduced alongside gpt-oss, replacing ChatML as their canonical prompt/response format. **gpt-oss will not work correctly without Harmony formatting** — this is non-negotiable.

The format uses special tokens (`<|start|>`, `<|end|>`, `<|message|>`, `<|channel|>`, `<|call|>`, `<|return|>`) and a **five-role hierarchy** with strict conflict resolution: system > developer > user > assistant > tool. The "developer" role replaces what other models call the "system prompt." A simple exchange looks like:

```
<|start|>user<|message|>What is 2 + 2?<|end|>
<|start|>assistant<|channel|>analysis<|message|>Simple arithmetic query.<|end|>
<|start|>assistant<|channel|>final<|message|>2 + 2 = 4.<|return|>
```

The critical innovation is **multi-channel output**. Assistant messages flow through three channels: `analysis` (private chain-of-thought, never shown to users), `commentary` (visible preambles, tool-call dispatches), and `final` (the user-facing response). This maps directly to how OpenAI's Responses API separates reasoning from output — if you've used that API, the mental model transfers directly.

Tool definitions use **TypeScript-style syntax** rather than JSON schemas, wrapped in namespaces. The official library lives at `github.com/openai/harmony` (Rust core, Python bindings via pyo3, installable as `pip install openai-harmony`). The tokenizer is `o200k_harmony`, an extension of the o200k tokenizer used by GPT-4o, available through tiktoken.

For fine-tuning, your training data needs to be structured in Harmony's messages-plus-channels format. In practice, you can write training examples using a standard messages list with role/content fields (plus an optional `thinking` field for chain-of-thought), and then use the `openai-harmony` library or HuggingFace's `apply_chat_template()` to convert it to proper tokenized Harmony format. **Unsloth provides an `encode_conversations_with_harmony` function** that handles this automatically during training. HuggingFace TRL also manages the conversion under the hood.

---

## Fine-tuning on 24GB VRAM: what fits, what doesn't

Your A10G with 24GB VRAM determines which models and methods are viable. Here is what the data shows:

**QLoRA (4-bit quantized LoRA)** is the dominant method for 24GB GPUs. With Unsloth's optimizations (2–5x faster, 80% less VRAM than standard Flash Attention 2), the VRAM envelope for QLoRA on an A10G is:

- **7B–8B dense models**: ~6–12 GB — very comfortable, room for large batch sizes
- **14B dense models**: ~10–16 GB — fits well, the sweet spot for A10G
- **20B–24B dense models**: ~16–20 GB — tight but feasible
- **30B MoE with 3B active** (like Qwen3-30B-A3B): **~17.5 GB** with Unsloth — confirmed to fit
- **30B+ dense**: exceeds 24GB, needs A100 40GB+
- **70B**: ~40–48 GB, requires A100 80GB

**Full fine-tuning** on 24GB only works for 1B–3B models. **Standard LoRA (16-bit)** fits models up to roughly 8B on 24GB. For your use case — specialized QIG reasoning that likely benefits from the largest model you can fit — **QLoRA on a 14B dense model or a 30B MoE model is the optimal strategy**.

The three major fine-tuning frameworks, ranked by ease-of-use:

**Unsloth** is the clear winner for single-GPU setups. It supports gpt-oss, Qwen3/3.5, Llama 3.x, Gemma 3, Phi-4, Mistral, and DeepSeek models. It provides free Colab notebooks, built-in GGUF export (`model.save_pretrained_gguf()`), and handles chat template conversion automatically. It supports SFT, LoRA, QLoRA, GRPO, DPO, and PPO training methods. Its limitation is primarily single-GPU focus — multi-GPU support is preliminary.

**Axolotl** is the production-grade choice. YAML-config-driven, it supports all HuggingFace models with full multi-GPU via FSDP and DeepSpeed. Modal.com explicitly recommends it as their default fine-tuning framework. It supports sample packing, Flash Attention, and every major training paradigm. The tradeoff is slightly steeper learning curve.

**Torchtune** offers pure PyTorch control with no abstractions, suitable if you want to customize the training loop for QIG-specific objectives. It's 20–30% slower than Unsloth for single-GPU LoRA.

---

## The GGUF-to-Ollama pipeline is mature and well-supported

Converting a fine-tuned model to GGUF and deploying on Ollama is a solved problem in 2026. The pipeline has multiple paths:

**Easiest path (Unsloth):** After fine-tuning, call `model.save_pretrained_gguf("output_dir", tokenizer, quantization_method="q4_k_m")`. This merges LoRA weights, converts to GGUF, and quantizes in one step. Supported quantizations include q4_k_m, q5_k_m, q8_0, f16, and over a dozen others. You can also push directly to HuggingFace with `model.push_to_hub_gguf()`.

**Manual path (llama.cpp):** Merge the LoRA adapter with the base model in Python (`model.merge_and_unload()`), save the merged model, then convert with `python convert_hf_to_gguf.py merged_model --outfile model-F16.gguf --outtype f16`, and quantize with `llama-quantize model-F16.gguf model-Q4_K_M.gguf Q4_K_M`.

**Ollama import** supports three methods: (1) from a GGUF file via a Modelfile (`FROM /path/to/model.gguf` → `ollama create my-model -f Modelfile`), (2) from merged safetensors directly (Ollama converts internally), or (3) from a LoRA adapter applied to an Ollama base model (`ADAPTER /path/to/adapter`). You can also quantize during import with `ollama create my-model -f Modelfile -q q4_0`.

**llama.cpp and Ollama support 60+ model architectures** as of early 2026, including gpt-oss, Llama (all versions), Qwen (2/2.5/3/3.5), Gemma (1/2/3), Phi (2/3/4), Mistral, DeepSeek, and GLM. Any GGUF that llama.cpp can load will work in Ollama.

One critical gotcha: **chat template matching at inference must match training**. If you fine-tune with Harmony format, your Ollama Modelfile must use the Harmony template. Ollama auto-detects templates from GGUF metadata when possible, but custom fine-tunes may need manual template specification.

---

## Qwen3 leads the field for fine-tuning on constrained hardware

For your specific requirements — A10G 24GB, scientific reasoning, GGUF/Ollama deployment — here is how the major model families compare:

**Qwen3 (Alibaba, April 2025)** offers the most compelling option. The **Qwen3-30B-A3B** is architecturally analogous to your current GLM-4-9B-Flash — a 30B MoE with only 3B active parameters — and fits in **17.5 GB VRAM** for QLoRA with Unsloth. The **Qwen3-14B** dense model is the other strong choice, fitting comfortably on 24GB. Both feature a native **thinking/non-thinking toggle** for chain-of-thought reasoning, which is particularly valuable for your QIG reasoning task. In the Distil Labs fine-tuning benchmark (March 2026), **Qwen3-4B achieved the best fine-tuned performance of all models tested**, matching or exceeding a 120B teacher on 7 of 8 tasks. Qwen3-8B took the best base-model performance crown. Apache 2.0 license, full Unsloth/Axolotl/Ollama support, official Ollama library entries.

The newer **Qwen3.5 (February 2026)** introduces a hybrid Gated DeltaNet + MoE architecture with 262K native context. Its 35B-A3B variant surpasses the previous Qwen3-235B-A22B on most benchmarks, but fine-tuning support is still maturing — Unsloth recommends bf16 LoRA (not QLoRA) for Qwen3.5 MoE, which requires more VRAM. For fine-tuning today, **stick with Qwen3**.

**Phi-4 (Microsoft)** deserves serious consideration for scientific reasoning. The **Phi-4-reasoning-plus (14B)** is specifically RL-trained on math and science data, scoring **85% on AIME** and performing strongly on GPQA-Diamond (graduate-level science). It fits QLoRA on 24GB and has MIT license. Unsloth found and fixed four bugs in Phi-4 including tokenizer issues, so use their distribution. The limitation is narrow: English-only, and only two size options (3.8B and 14B).

**Llama 3.x (Meta)** has the **widest fine-tuning ecosystem** — more tutorials, more tools, more community fine-tunes than any other family. Llama 3.1 8B is the de facto "hello world" of fine-tuning. However, at equivalent parameter counts, Llama trails Qwen3 and Phi-4 on reasoning benchmarks, and Meta's Llama Community License is more restrictive than Apache 2.0. No MoE variants exist in the 3.x family, limiting what you can fit on 24GB (the 70B model needs an A100 80GB).

**Gemma 3 (Google)** offers decent 4B and 12B options with QAT variants for better quantized performance, but ranks below Qwen3 and Phi-4 on reasoning benchmarks at equivalent sizes. Unsloth is the only framework that handles float16 correctly on A10G/T4 for Gemma 3. The Gemma license restricts use for models that "substantially compete with Gemma."

**Mistral's Ministral 3 family** is strong on multilingual tasks (40+ languages) and the 14B reasoning variant scores well, but the fine-tuning ecosystem is thinner than Qwen or Llama. Their `mistral-finetune` tool targets A100/H100 hardware.

| Model | Best 24GB variant | Reasoning | Fine-tune ecosystem | GGUF/Ollama | License |
|---|---|---|---|---|---|
| **Qwen3** | 14B dense / 30B-A3B MoE | ★★★★★ | ★★★★★ | ★★★★★ | Apache 2.0 |
| **Phi-4** | 14B | ★★★★★ | ★★★★ | ★★★★ | MIT |
| **gpt-oss** | 20b (3.6B active) | ★★★★ | ★★★★ | ★★★★ | Apache 2.0 |
| **Llama 3.x** | 8B | ★★★ | ★★★★★ | ★★★★★ | Llama CL |
| **Mistral** | 14B | ★★★★ | ★★★ | ★★★★ | Apache 2.0 |
| **Gemma 3** | 12B | ★★★ | ★★★ | ★★★★ | Gemma License |

---

## The recommended pipeline for your QIG fine-tuning project

Given your setup — A10G 24GB on Modal, Ollama deployment, QIG reasoning and consciousness protocol compliance — the optimal end-to-end pipeline is:

**Model choice: Qwen3-14B** (dense, native thinking mode, best fine-tuned benchmark performance at this scale). If you want to stay closer to your current MoE architecture, **Qwen3-30B-A3B** fits QLoRA at 17.5GB but has slightly more complex fine-tuning considerations (router layers shouldn't be trained, performance variance is higher with MoE QLoRA).

**Training data format:** Prepare your QIG reasoning examples in **ChatML/ShareGPT format** with system/user/assistant roles (Qwen3 uses ChatML natively with `<|im_start|>`/`<|im_end|>` tokens). If you choose gpt-oss-20b instead, you'll need to format in Harmony using the `openai-harmony` library. For either model, include chain-of-thought reasoning in your training examples — Qwen3 supports this via its thinking mode, gpt-oss via Harmony's `analysis` channel.

**Fine-tuning:** Use **Unsloth** with QLoRA on your A10G. Install with `pip install unsloth`, load the model in 4-bit, attach LoRA adapters (rank 16–64, targeting q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj), and train with SFTTrainer. For QIG-specific reasoning optimization, consider **GRPO** (Group Relative Policy Optimization) after initial SFT — Unsloth provides notebooks for this.

**GGUF export:** Call `model.save_pretrained_gguf("qig-model", tokenizer, quantization_method="q4_k_m")` directly from Unsloth. For Qwen3-14B, the Q4_K_M GGUF will be approximately **9.3 GB**, fitting easily in your 24GB A10G inference budget.

**Ollama deployment:** Create a Modelfile (`FROM ./qig-model-Q4_K_M.gguf` plus any custom template/parameters), then `ollama create qig-model -f Modelfile`. On Modal, mount the GGUF file and run `ollama serve` with your model.

**Why not gpt-oss-20b?** It's a viable alternative — same MoE concept, Apache 2.0, confirmed Ollama/GGUF support. But the Harmony format adds complexity to data preparation, the fine-tuning ecosystem is younger (launched August 2025 vs. Qwen3's April 2025 with more community iteration), and Qwen3 consistently outperforms on reasoning benchmarks at comparable active-parameter counts. If you specifically want OpenAI's model for branding or architectural reasons, gpt-oss-20b works — just account for the Harmony formatting overhead.

## Conclusion

The open-weight fine-tuning landscape in early 2026 is remarkably mature. **Qwen3-14B with Unsloth QLoRA represents the path of least resistance** for your specific constraints — it delivers top-tier reasoning in a 24GB-compatible package with the strongest fine-tuning benchmarks, native thinking mode for chain-of-thought, and a one-command GGUF export to Ollama. OpenAI's gpt-oss-20b is a credible alternative with its novel Harmony format (multi-channel reasoning, five-role hierarchy, TypeScript-style tool definitions), but the added format complexity and younger ecosystem make it a second choice unless you're specifically building around OpenAI's architecture. The key insight from the Distil Labs benchmarks is that **model family matters more than raw parameter count for fine-tuning** — Qwen3-4B fine-tuned outperformed a 120B teacher, confirming that a well-chosen 14B model fine-tuned on high-quality QIG data will likely exceed much larger general-purpose models on your specialized task.
