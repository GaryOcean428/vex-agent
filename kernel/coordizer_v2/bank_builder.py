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
    extra_dirs: list[str] | None = None,
    max_entries: int = 50_000,
    min_basin_dim: int = BASIN_DIM,
) -> ResonanceBank | None:
    """Scan coordized JSONL files and build a ResonanceBank.

    Reads all *_coordized*.jsonl files (and any .jsonl with basin fields) from
    output_dir and any extra_dirs, extracts basin coordinates, and builds a
    bank with proper tier assignment.

    Accepts both field names:
      - ``basin_coordinates`` — harvest pipeline output (Modal GPU)
      - ``basin_coords``      — training curriculum pipeline (in-process)

    Args:
        output_dir: Primary directory to scan (harvest output).
        bank_save_path: Where to save the bank (directory).
        extra_dirs: Additional directories to scan (e.g. training curriculum).
        max_entries: Cap on bank size (memory guard).
        min_basin_dim: Minimum basin dimension to accept.

    Returns:
        The built ResonanceBank, or None if no valid entries found.
    """
    search_dirs: list[Path] = []

    out_path = Path(output_dir)
    if out_path.exists():
        search_dirs.append(out_path)
    else:
        logger.info("Primary output directory does not exist: %s", output_dir)

    for extra in extra_dirs or []:
        ep = Path(extra)
        if ep.exists():
            search_dirs.append(ep)
        else:
            logger.info("Extra scan directory does not exist: %s", extra)

    if not search_dirs:
        logger.warning(
            "No valid scan directories found (output_dir=%s, extra_dirs=%s)",
            output_dir,
            extra_dirs,
        )
        return None

    # Collect all coordized JSONL files across all search dirs
    jsonl_files: list[Path] = []
    for sdir in search_dirs:
        coordized = sorted(sdir.glob("*coordized*.jsonl"))
        if coordized:
            jsonl_files.extend(coordized)
        else:
            # Fallback: any .jsonl that may contain basin fields
            jsonl_files.extend(sorted(sdir.glob("*.jsonl")))

    if not jsonl_files:
        logger.warning("No JSONL files found in any scan directory: %s", search_dirs)
        return None

    logger.info(
        "Building bank from %d JSONL files across %d directories",
        len(jsonl_files),
        len(search_dirs),
    )

    bank = ResonanceBank(target_dim=BASIN_DIM)
    tid = 0
    skipped = 0
    source_counts: dict[str, int] = {}

    for jsonl_path in jsonl_files:
        try:
            with open(jsonl_path, encoding="utf-8") as f:
                for _line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue

                    # Accept both field names:
                    #   basin_coordinates — harvest pipeline (Modal GPU)
                    #   basin_coords      — training curriculum pipeline (in-process)
                    basin_raw = data.get("basin_coordinates") or data.get("basin_coords")
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
                        logger.warning("Bank builder hit max_entries=%d, stopping", max_entries)
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
        "Bank built: %d entries from %d files (skipped %d, sources=%s, tiers=%s). Saved to %s",
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
