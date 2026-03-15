"""
Bank Builder — Build ResonanceBank from coordized JSONL output
==============================================================

THE MISSING LINK: The harvest pipeline writes coordized JSONL with
basin_coordinates per entry, but nobody reads those back into a
ResonanceBank. This module closes that gap.

Scans coordized JSONL output directory, reads basin coordinates,
builds a ResonanceBank, and saves it to disk in the format that
ResonanceBank.from_file() expects (bank_coordinates.npy +
bank_token_ids.npy + bank_meta.json).

Called by HarvestScheduler after successful file processing, or
manually via rebuild_bank_from_output().

Zero Euclidean contamination. Fisher-Rao is the ONLY distance metric.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import BASIN_DIM, to_simplex
from .resonance_bank import ResonanceBank
from .types import HarmonicTier

logger = logging.getLogger(__name__)


def rebuild_bank_from_output(
    output_dir: str,
    bank_save_path: str,
    *,
    max_entries: int = 50_000,
    min_basin_dim: int = BASIN_DIM,
) -> ResonanceBank | None:
    """Scan coordized JSONL files and build a ResonanceBank.

    Reads all *_coordized*.jsonl files from output_dir, extracts
    basin_coordinates, and builds a bank with proper tier assignment.

    Args:
        output_dir: Directory containing coordized JSONL files.
        bank_save_path: Where to save the bank (directory).
        max_entries: Cap on bank size (memory guard).
        min_basin_dim: Minimum basin dimension to accept.

    Returns:
        The built ResonanceBank, or None if no valid entries found.
    """
    out_path = Path(output_dir)
    if not out_path.exists():
        logger.warning("Output directory does not exist: %s", output_dir)
        return None

    # Collect all coordized JSONL files
    jsonl_files = sorted(out_path.glob("*coordized*.jsonl"))
    if not jsonl_files:
        # Also check for any .jsonl with basin_coordinates
        jsonl_files = sorted(out_path.glob("*.jsonl"))

    if not jsonl_files:
        logger.warning("No JSONL files found in %s", output_dir)
        return None

    logger.info("Building bank from %d JSONL files in %s", len(jsonl_files), output_dir)

    bank = ResonanceBank(target_dim=BASIN_DIM)
    tid = 0
    skipped = 0
    source_counts: dict[str, int] = {}

    for jsonl_path in jsonl_files:
        try:
            with open(jsonl_path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue

                    basin_raw = data.get("basin_coordinates")
                    if basin_raw is None or not isinstance(basin_raw, list):
                        skipped += 1
                        continue

                    if len(basin_raw) < min_basin_dim:
                        skipped += 1
                        continue

                    # Project to simplex (safety — should already be on Δ⁶³)
                    basin = to_simplex(np.array(basin_raw[:BASIN_DIM], dtype=np.float64))

                    # Use first 80 chars of text as token string
                    text = data.get("text", "")
                    token_str = text[:80].replace("\n", " ").strip() or f"<chunk_{tid}>"

                    # Determine tier from source
                    source = data.get("source", "document")
                    source_counts[source] = source_counts.get(source, 0) + 1

                    if source == "curriculum":
                        tier = HarmonicTier.FUNDAMENTAL
                    elif source in ("foraging", "llm_cogeneration"):
                        tier = HarmonicTier.FIRST_HARMONIC
                    elif source == "conversation":
                        tier = HarmonicTier.UPPER_HARMONIC
                    else:
                        tier = HarmonicTier.OVERTONE_HAZE

                    bank.coordinates[tid] = basin
                    bank.token_strings[tid] = token_str
                    bank.tiers[tid] = tier
                    bank.activation_counts[tid] = 0
                    bank.origin[tid] = "harvested"
                    bank._bank_total_count += 1
                    tid += 1

                    if tid >= max_entries:
                        logger.warning(
                            "Bank builder hit max_entries=%d, stopping", max_entries
                        )
                        break

        except Exception as e:
            logger.error("Error reading %s: %s", jsonl_path, e)
            continue

        if tid >= max_entries:
            break

    if tid == 0:
        logger.warning("No valid basin coordinates found in any JSONL file")
        return None

    # Assign frequencies and basin mass from entropy
    bank._assign_tiers()
    bank._assign_frequencies()
    bank._rebuild_matrix()

    # Save to disk
    bank.save(bank_save_path)

    tier_dist = bank.tier_distribution()
    logger.info(
        "Bank built: %d entries from %d files "
        "(skipped %d, sources=%s, tiers=%s). Saved to %s",
        tid,
        len(jsonl_files),
        skipped,
        source_counts,
        tier_dist,
        bank_save_path,
    )

    return bank


def get_bank_stats(bank_path: str) -> dict[str, Any] | None:
    """Get stats from a saved bank without loading the full bank."""
    bp = Path(bank_path)
    meta_path = bp / "bank_meta.json"
    if not meta_path.exists():
        return None

    try:
        with open(meta_path) as f:
            meta = json.load(f)
        return {
            "n_tokens": meta.get("n_tokens", 0),
            "dim": meta.get("dim", 0),
            "bank_lived_count": meta.get("bank_lived_count", 0),
            "bank_total_count": meta.get("bank_total_count", 0),
            "has_coordinates": (bp / "bank_coordinates.npy").exists(),
            "has_token_ids": (bp / "bank_token_ids.npy").exists(),
        }
    except Exception as e:
        logger.error("Error reading bank stats: %s", e)
        return None
