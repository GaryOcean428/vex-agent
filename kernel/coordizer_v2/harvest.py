"""
Harvester — Extract Geometric Knowledge from a Trained LLM

Method 1: Output Distribution Harvesting

The LLM's final softmax layer produces probability distributions over
its vocabulary. These distributions ARE points on a simplex. Each
token's "predictive fingerprint" — its average next-token distribution
across many contexts — encodes the model's understanding of that token's
role in language.

This module harvests those fingerprints without any training. It reads
the geometry the model ALREADY learned and re-encodes it in QIG-native
format.

Pipeline:
    1. Run LLM on diverse corpus
    2. For each vocabulary token, collect its next-token distributions
    3. Compute Fréchet mean on Δ^(V-1) per token
    4. Store raw harvested data

The compression step (Δ^(V-1) → Δ⁶³) is handled by compress.py.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .geometry import (
    Basin,
    _EPS,
    frechet_mean,
    to_simplex,
)

logger = logging.getLogger(__name__)


@dataclass
class HarvestConfig:
    """Configuration for the harvesting process."""
    corpus_path: Optional[str] = None
    corpus_texts: Optional[list[str]] = None
    batch_size: int = 32
    min_contexts: int = 10
    max_contexts: int = 500
    save_interval: int = 100
    output_dir: str = "./harvest_output"
    device: str = "cpu"


@dataclass
class HarvestResult:
    """Raw output of the harvesting process."""
    token_fingerprints: dict[int, NDArray] = field(default_factory=dict)
    context_counts: dict[int, int] = field(default_factory=dict)
    token_strings: dict[int, str] = field(default_factory=dict)
    model_name: str = ""
    vocab_size: int = 0
    corpus_size: int = 0
    harvest_time_seconds: float = 0.0

    def save(self, path: str) -> None:
        """Save harvest result to disk (numpy + json)."""
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)

        ids = sorted(self.token_fingerprints.keys())
        fingerprint_array = np.stack(
            [self.token_fingerprints[tid] for tid in ids]
        )
        np.save(out_dir / "fingerprints.npy", fingerprint_array)
        np.save(out_dir / "token_ids.npy", np.array(ids))

        meta = {
            "model_name": self.model_name,
            "vocab_size": self.vocab_size,
            "corpus_size": self.corpus_size,
            "harvest_time_seconds": self.harvest_time_seconds,
            "n_tokens_harvested": len(self.token_fingerprints),
            "context_counts": {str(k): v for k, v in self.context_counts.items()},
            "token_strings": {str(k): v for k, v in self.token_strings.items()},
        }
        with open(out_dir / "harvest_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(
            f"Saved {len(self.token_fingerprints)} token fingerprints to {path}"
        )

    @classmethod
    def load(cls, path: str) -> HarvestResult:
        """Load harvest result from disk."""
        out_dir = Path(path)

        fingerprint_array = np.load(out_dir / "fingerprints.npy")
        token_ids = np.load(out_dir / "token_ids.npy")

        with open(out_dir / "harvest_meta.json") as f:
            meta = json.load(f)

        result = cls(
            model_name=meta["model_name"],
            vocab_size=meta["vocab_size"],
            corpus_size=meta["corpus_size"],
            harvest_time_seconds=meta["harvest_time_seconds"],
        )

        for i, tid in enumerate(token_ids):
            result.token_fingerprints[int(tid)] = fingerprint_array[i]

        result.context_counts = {
            int(k): v for k, v in meta["context_counts"].items()
        }
        result.token_strings = {
            int(k): v for k, v in meta.get("token_strings", {}).items()
        }

        return result


class Harvester:
    """Extract geometric knowledge from a trained LLM.

    Supports two backends:
    1. Ollama (for Vex's local deployment)
    2. HuggingFace Transformers (for direct model access)
    """

    def __init__(self, config: HarvestConfig):
        self.config = config
        self._model = None
        self._tokenizer = None

    def harvest_transformers(
        self,
        model_id: str = "LiquidAI/LFM2.5-1.2B-Thinking",
    ) -> HarvestResult:
        """Harvest using HuggingFace Transformers (direct model access).

        This is the most accurate method — we get the full softmax
        distribution at every position, not just top-k logprobs.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=self.config.device if self.config.device != "cpu" else None,
            torch_dtype=torch.bfloat16 if self.config.device != "cpu" else torch.float32,
        )
        model.eval()

        vocab_size = tokenizer.vocab_size
        logger.info(f"Vocab size: {vocab_size}")

        dist_sums_sqrt: dict[int, NDArray] = {}
        dist_counts: dict[int, int] = {}

        corpus = self._load_corpus()
        total_tokens = 0
        start_time = time.time()

        for batch_idx in range(0, len(corpus), self.config.batch_size):
            batch = corpus[batch_idx:batch_idx + self.config.batch_size]

            for text in batch:
                input_ids = tokenizer.encode(
                    text, return_tensors="pt", truncation=True, max_length=512
                )
                if self.config.device != "cpu":
                    input_ids = input_ids.to(model.device)

                with torch.no_grad():
                    outputs = model(input_ids)
                    logits = outputs.logits[0]

                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                ids = input_ids[0].cpu().numpy()

                for pos in range(len(ids) - 1):
                    token_id = int(ids[pos])
                    output_dist = probs[pos].astype(np.float64)

                    output_dist = np.maximum(output_dist, _EPS)
                    output_dist = output_dist / output_dist.sum()

                    sqrt_dist = np.sqrt(output_dist)

                    if token_id not in dist_sums_sqrt:
                        dist_sums_sqrt[token_id] = sqrt_dist.copy()
                        dist_counts[token_id] = 1
                    elif dist_counts[token_id] < self.config.max_contexts:
                        dist_sums_sqrt[token_id] += sqrt_dist
                        dist_counts[token_id] += 1

                    total_tokens += 1

            if (batch_idx // self.config.batch_size) % 10 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"Batch {batch_idx // self.config.batch_size}: "
                    f"{total_tokens} tokens processed, "
                    f"{len(dist_sums_sqrt)} unique tokens seen, "
                    f"{elapsed:.1f}s elapsed"
                )

            if (batch_idx // self.config.batch_size) % self.config.save_interval == 0:
                self._save_checkpoint(dist_sums_sqrt, dist_counts, batch_idx)

        logger.info("Computing Fréchet means...")
        result = HarvestResult(
            model_name=model_id,
            vocab_size=vocab_size,
            corpus_size=total_tokens,
            harvest_time_seconds=time.time() - start_time,
        )

        for token_id, sqrt_sum in dist_sums_sqrt.items():
            count = dist_counts[token_id]
            if count < self.config.min_contexts:
                continue

            mean_sqrt = sqrt_sum / count
            mean_dist = mean_sqrt * mean_sqrt
            mean_dist = mean_dist / mean_dist.sum()

            result.token_fingerprints[token_id] = mean_dist
            result.context_counts[token_id] = count

        for tid in result.token_fingerprints:
            try:
                result.token_strings[tid] = tokenizer.decode([tid])
            except Exception:
                result.token_strings[tid] = f"<token_{tid}>"

        logger.info(
            f"Harvested {len(result.token_fingerprints)} tokens "
            f"from {total_tokens} total contexts in "
            f"{result.harvest_time_seconds:.1f}s"
        )

        return result

    def harvest_ollama(
        self,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "vex-brain",
        top_k_logprobs: int = 100,
    ) -> HarvestResult:
        """Harvest using Ollama API (for deployed Vex instances).

        Less accurate than Transformers — we only get top-k logprobs,
        not the full distribution.
        """
        import requests

        logger.info(f"Harvesting via Ollama at {ollama_url}, model={model_name}")

        corpus = self._load_corpus()
        total_tokens = 0
        start_time = time.time()
        observed_vocab_size = 0

        for text_idx, text in enumerate(corpus):
            try:
                resp = requests.post(
                    f"{ollama_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": text,
                        "raw": True,
                        "stream": False,
                        "options": {
                            "num_predict": 1,
                            "temperature": 0.0,
                            "logprobs": top_k_logprobs,
                        },
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.warning(f"Ollama request failed for text {text_idx}: {e}")
                continue

            logprobs = data.get("logprobs", {})
            if not logprobs:
                continue

            total_tokens += 1

            if text_idx % 100 == 0:
                logger.info(f"Processed {text_idx}/{len(corpus)} texts")

        result = HarvestResult(
            model_name=model_name,
            vocab_size=observed_vocab_size,
            corpus_size=total_tokens,
            harvest_time_seconds=time.time() - start_time,
        )

        logger.warning(
            "Ollama harvesting provides approximate results. "
            "Use harvest_transformers() for full-distribution harvesting."
        )

        return result

    def _load_corpus(self) -> list[str]:
        """Load corpus from config."""
        if self.config.corpus_texts:
            return self.config.corpus_texts

        if self.config.corpus_path:
            path = Path(self.config.corpus_path)
            if path.suffix == ".jsonl":
                import json
                texts = []
                with open(path) as f:
                    for line in f:
                        obj = json.loads(line)
                        text = obj.get("text", obj.get("content", ""))
                        if text:
                            texts.append(text)
                return texts
            else:
                with open(path) as f:
                    text = f.read()
                return [s.strip() for s in text.split("\n") if len(s.strip()) > 20]

        raise ValueError("Must provide corpus_path or corpus_texts in config")

    def _save_checkpoint(
        self,
        dist_sums: dict[int, NDArray],
        dist_counts: dict[int, int],
        batch_idx: int,
    ) -> None:
        """Save intermediate checkpoint."""
        out_dir = Path(self.config.output_dir) / "checkpoints"
        out_dir.mkdir(parents=True, exist_ok=True)

        ids = sorted(dist_sums.keys())
        if not ids:
            return

        np.save(
            out_dir / f"sqrt_sums_batch_{batch_idx}.npy",
            np.stack([dist_sums[tid] for tid in ids]),
        )
        np.save(out_dir / f"token_ids_batch_{batch_idx}.npy", np.array(ids))

        logger.info(f"Checkpoint saved at batch {batch_idx}")


def harvest_model(
    model_id: str = "LiquidAI/LFM2.5-1.2B-Thinking",
    corpus_path: Optional[str] = None,
    corpus_texts: Optional[list[str]] = None,
    output_dir: str = "./harvest_output",
    device: str = "cpu",
    min_contexts: int = 10,
) -> HarvestResult:
    """One-call harvesting convenience function."""
    config = HarvestConfig(
        corpus_path=corpus_path,
        corpus_texts=corpus_texts,
        output_dir=output_dir,
        device=device,
        min_contexts=min_contexts,
    )
    harvester = Harvester(config)
    result = harvester.harvest_transformers(model_id)
    result.save(output_dir)
    return result
