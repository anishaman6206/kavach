"""
scripts/build_dataset.py
=========================
Multi-source dataset assembly pipeline for Kavach MuRIL fine-tuning.

Merges four sources into one clean CSV with a consistent schema:
  1. data/final_scam_dataset.csv     — existing Kavach dataset (no speaker tags)
  2. menaattia/phone-scam-dataset    — HuggingFace, caller:/receiver: tags
  3. BothBosu/multi-agent-scam-conversation — HuggingFace, Suspect:/Innocent: tags
  4. shakeleoatmeal/phone-scam-detection-synthetic — HuggingFace, caller:/receiver: tags

Output schema (one row = one conversation):
  conversation_id  : str  — unique ID: {source}_{index:05d}
  text             : str  — full conversation, CALLER:/USER: normalised
  label            : int  — 0=legitimate, 1=scam
  source           : str  — source identifier
  language         : str  — "en" | "hi" | "mixed"
  has_speaker_turns: bool — True if CALLER:/USER: tags present
  word_count       : int

Speaker normalisation:
  menaattia        : caller: -> CALLER:   receiver: -> USER:
  BothBosu         : Suspect: -> CALLER:  Innocent: -> USER:
  shakeleoatmeal   : caller: -> CALLER:   receiver: -> USER:
  kavach_original  : no tags — has_speaker_turns stays False

Quality filters:
  • Remove rows with word_count < 50
  • Remove exact duplicates on normalised text

Usage:
  python scripts/build_dataset.py
  python scripts/build_dataset.py --output-dir data/
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# -- sys.path so kavach package is importable ----------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

MIN_WORDS     = 50
SOURCE_KAVACH = "kavach_original"
SOURCE_MENA   = "menaattia"
SOURCE_BOTH   = "bothbosu"
SOURCE_SHAKE  = "shakeleoatmeal"


# -----------------------------------------------------------------------------
# Text helpers
# -----------------------------------------------------------------------------

def _normalise_speakers_caller_receiver(text: str) -> str:
    """caller: -> CALLER:   receiver: -> USER:  (case-insensitive)"""
    text = re.sub(r"(?i)\bcaller\s*:", "CALLER:", text)
    text = re.sub(r"(?i)\breceiver\s*:", "USER:", text)
    return text


def _normalise_speakers_bothbosu(text: str) -> str:
    """Suspect: -> CALLER:   Innocent: -> USER:"""
    text = re.sub(r"(?i)\bSuspect\s*:", "CALLER:", text)
    text = re.sub(r"(?i)\bInnocent\s*:", "USER:", text)
    return text


def _clean_text(text: str) -> str:
    """Collapse whitespace, strip leading/trailing space."""
    text = text.strip()
    # Collapse multiple blank lines to one
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces/tabs within a line
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def _detect_language(text: str) -> str:
    """
    Heuristic language detection — no external library required.
    Checks for Devanagari Unicode block (U+0900–U+097F).
    """
    devanagari = sum(1 for ch in text if "\u0900" <= ch <= "\u097F")
    total_alpha = sum(1 for ch in text if ch.isalpha())
    if total_alpha == 0:
        return "en"
    ratio = devanagari / total_alpha
    if ratio > 0.40:
        return "hi"
    if ratio > 0.05:
        return "mixed"
    return "en"


def _word_count(text: str) -> int:
    return len(text.split())


def _has_speaker_turns(text: str) -> bool:
    """True if the normalised text contains CALLER: or USER: markers."""
    return bool(re.search(r"\b(CALLER|USER)\s*:", text))


# -----------------------------------------------------------------------------
# Per-source loaders — each returns a list of dicts (pre-schema)
# -----------------------------------------------------------------------------

def _load_kavach_original(csv_path: str) -> list[dict]:
    """
    Existing data/final_scam_dataset.csv — columns: text, label.
    No speaker tags. Language is English (no Devanagari in this dataset).
    """
    print(f"[kavach_original] Loading from {csv_path} ...")
    df = pd.read_csv(csv_path)
    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected columns {required}, got {list(df.columns)}")

    rows = []
    for idx, row in df.iterrows():
        text = _clean_text(str(row["text"]))
        rows.append({
            "text":  text,
            "label": int(row["label"]),
        })
    print(f"[kavach_original] Loaded {len(rows)} rows.")
    return rows


def _load_hf_all_splits(hf_id: str):
    """Load all splits of a HuggingFace dataset and concatenate."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] 'datasets' library not installed. Run: pip install datasets")
        sys.exit(1)

    ds = load_dataset(hf_id)
    combined = []
    for split_name, split in ds.items():
        combined.extend(split)
    return combined


def _load_menaattia() -> list[dict]:
    """
    menaattia/phone-scam-dataset — columns: dialogue, label.
    Speaker tags: caller: -> CALLER:, receiver: -> USER:
    """
    print("[menaattia] Loading from HuggingFace ...")
    rows_raw = _load_hf_all_splits("menaattia/phone-scam-dataset")
    rows = []
    for row in rows_raw:
        text = _normalise_speakers_caller_receiver(str(row["dialogue"]))
        text = _clean_text(text)
        rows.append({
            "text":  text,
            "label": int(row["label"]),
        })
    print(f"[menaattia] Loaded {len(rows)} rows.")
    return rows


def _load_bothbosu() -> list[dict]:
    """
    BothBosu/multi-agent-scam-conversation — columns: dialogue, labels, personality, type.
    Speaker tags: Suspect: -> CALLER:, Innocent: -> USER:
    Label column is 'labels' (not 'label').
    Use labels==1 as scam, labels==0 as legit.
    """
    print("[bothbosu] Loading from HuggingFace ...")
    rows_raw = _load_hf_all_splits("BothBosu/multi-agent-scam-conversation")
    rows = []
    for row in rows_raw:
        text = _normalise_speakers_bothbosu(str(row["dialogue"]))
        text = _clean_text(text)
        rows.append({
            "text":  text,
            "label": int(row["labels"]),   # NOTE: 'labels', not 'label'
        })
    print(f"[bothbosu] Loaded {len(rows)} rows.")
    return rows


def _load_shakeleoatmeal() -> list[dict]:
    """
    shakeleoatmeal/phone-scam-detection-synthetic -- columns: dialogue, label, type, ...
    Speaker tags: caller: -> CALLER:, receiver: -> USER:
    """
    print("[shakeleoatmeal] Loading from HuggingFace ...")
    rows_raw = _load_hf_all_splits("shakeleoatmeal/phone-scam-detection-synthetic")
    rows = []
    for row in rows_raw:
        text = _normalise_speakers_caller_receiver(str(row["dialogue"]))
        text = _clean_text(text)
        rows.append({
            "text":  text,
            "label": int(row["label"]),
        })
    print(f"[shakeleoatmeal] Loaded {len(rows)} rows.")
    return rows


# -----------------------------------------------------------------------------
# Assembly
# -----------------------------------------------------------------------------

def _to_dataframe(rows: list[dict], source: str) -> pd.DataFrame:
    """
    Convert a list of {text, label} dicts to a fully-featured DataFrame
    with conversation_id, source, language, has_speaker_turns, word_count.
    """
    records = []
    for idx, row in enumerate(rows):
        text  = row["text"]
        label = row["label"]
        records.append({
            "conversation_id":   f"{source}_{idx:05d}",
            "text":              text,
            "label":             label,
            "source":            source,
            "language":          _detect_language(text),
            "has_speaker_turns": _has_speaker_turns(text),
            "word_count":        _word_count(text),
        })
    return pd.DataFrame(records)


def build_dataset(
    kavach_csv: str,
    output_dir: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Main assembly function. Returns (merged_df, stats_dict).

    Args:
        kavach_csv : path to data/final_scam_dataset.csv
        output_dir : directory to write dataset_merged.csv and dataset_report.txt
    """
    # -- Load all sources ------------------------------------------------------
    kavach_rows = _load_kavach_original(kavach_csv)
    mena_rows   = _load_menaattia()
    both_rows   = _load_bothbosu()
    shake_rows  = _load_shakeleoatmeal()

    # -- Convert to DataFrames with full schema --------------------------------
    dfs = {
        SOURCE_KAVACH: _to_dataframe(kavach_rows, SOURCE_KAVACH),
        SOURCE_MENA:   _to_dataframe(mena_rows,   SOURCE_MENA),
        SOURCE_BOTH:   _to_dataframe(both_rows,   SOURCE_BOTH),
        SOURCE_SHAKE:  _to_dataframe(shake_rows,  SOURCE_SHAKE),
    }

    raw_counts = {src: len(df) for src, df in dfs.items()}
    total_raw = sum(raw_counts.values())
    print(f"\n[assembly] Raw totals before filtering: {raw_counts} -> {total_raw} rows")

    merged = pd.concat(list(dfs.values()), ignore_index=True)

    # -- Quality filter 1: minimum word count ---------------------------------
    before_wc = len(merged)
    merged = merged[merged["word_count"] >= MIN_WORDS].copy()
    removed_wc = before_wc - len(merged)
    print(f"[filter] word_count < {MIN_WORDS}: removed {removed_wc} rows")

    # -- Quality filter 2: exact deduplication on normalised text -------------
    before_dedup = len(merged)
    merged = merged.drop_duplicates(subset=["text"], keep="first").copy()
    removed_dedup = before_dedup - len(merged)
    print(f"[filter] exact duplicates: removed {removed_dedup} rows")

    merged = merged.reset_index(drop=True)
    print(f"[assembly] Final dataset: {len(merged)} rows\n")

    stats = {
        "raw_counts":    raw_counts,
        "total_raw":     total_raw,
        "removed_wc":    removed_wc,
        "removed_dedup": removed_dedup,
        "final_total":   len(merged),
    }

    return merged, stats


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------

def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "0.0%"


def build_report(df: pd.DataFrame, stats: dict) -> str:
    total = len(df)
    lines = []

    lines.append("=" * 60)
    lines.append("  KAVACH DATASET ASSEMBLY REPORT")
    lines.append("=" * 60)
    lines.append(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"  Final rows: {total:,}")
    lines.append("")

    # -- Per-source breakdown --------------------------------------------------
    lines.append("-" * 60)
    lines.append("  Per-source breakdown")
    lines.append("-" * 60)
    for src in [SOURCE_KAVACH, SOURCE_MENA, SOURCE_BOTH, SOURCE_SHAKE]:
        sub  = df[df["source"] == src]
        n    = len(sub)
        raw  = stats["raw_counts"].get(src, 0)
        scam = (sub["label"] == 1).sum()
        leg  = (sub["label"] == 0).sum()
        spk  = sub["has_speaker_turns"].sum()
        lines.append(
            f"  {src:<22}  {n:>5,} rows  ({_pct(n, total):>6})  "
            f"[raw: {raw:,}]"
        )
        lines.append(
            f"    Scam: {scam:,}  Legit: {leg:,}  "
            f"Speaker turns: {spk:,}/{n:,} ({_pct(int(spk), n)})"
        )
    lines.append("")

    # -- Overall label distribution --------------------------------------------
    lines.append("-" * 60)
    lines.append("  Label distribution (overall)")
    lines.append("-" * 60)
    scam_total = (df["label"] == 1).sum()
    leg_total  = (df["label"] == 0).sum()
    lines.append(f"  Scam  (1) : {scam_total:,}  ({_pct(scam_total, total)})")
    lines.append(f"  Legit (0) : {leg_total:,}   ({_pct(leg_total, total)})")
    lines.append("")

    # -- Language distribution -------------------------------------------------
    lines.append("-" * 60)
    lines.append("  Language distribution")
    lines.append("-" * 60)
    for lang, count in df["language"].value_counts().items():
        lines.append(f"  {lang:<8} : {count:>6,}  ({_pct(count, total)})")
    lines.append("")

    # -- Speaker turns ---------------------------------------------------------
    lines.append("-" * 60)
    lines.append("  Speaker turns")
    lines.append("-" * 60)
    has_spk = df["has_speaker_turns"].sum()
    no_spk  = total - has_spk
    lines.append(f"  Has CALLER:/USER: : {has_spk:>6,}  ({_pct(int(has_spk), total)})")
    lines.append(f"  No speaker tags   : {no_spk:>6,}  ({_pct(no_spk, total)})")
    lines.append("")

    # -- Word count stats ------------------------------------------------------
    lines.append("-" * 60)
    lines.append("  Word count statistics")
    lines.append("-" * 60)
    wc = df["word_count"]
    lines.append(f"  Mean   : {wc.mean():.1f}")
    lines.append(f"  Median : {wc.median():.1f}")
    lines.append(f"  Min    : {wc.min()}")
    lines.append(f"  Max    : {wc.max()}")
    lines.append(f"  P25    : {wc.quantile(0.25):.0f}")
    lines.append(f"  P75    : {wc.quantile(0.75):.0f}")
    lines.append("")

    # -- Quality filter summary ------------------------------------------------
    lines.append("-" * 60)
    lines.append("  Quality filters applied")
    lines.append("-" * 60)
    lines.append(f"  Raw total (pre-filter)  : {stats['total_raw']:,}")
    lines.append(f"  Removed (word_count<50) : {stats['removed_wc']:,}")
    lines.append(f"  Removed (exact dupes)   : {stats['removed_dedup']:,}")
    lines.append(f"  Final rows retained     : {stats['final_total']:,}")
    lines.append("")

    # -- MuRIL fine-tuning note ------------------------------------------------
    lines.append("-" * 60)
    lines.append("  MuRIL fine-tuning readiness")
    lines.append("-" * 60)
    lines.append("  Train/test split: NOT done here -- MuRILTrainer splits at")
    lines.append("  conversation_id level (group-aware). Run:")
    lines.append("    MuRILTrainer().train('data/dataset_merged.csv', epochs=3)")
    lines.append("=" * 60)

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble multi-source scam detection dataset."
    )
    parser.add_argument(
        "--kavach-csv",
        default=str(Path(__file__).resolve().parent.parent / "data" / "final_scam_dataset.csv"),
        help="Path to existing Kavach CSV (default: data/final_scam_dataset.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent.parent / "data"),
        help="Output directory for dataset_merged.csv and dataset_report.txt (default: data/)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.kavach_csv):
        print(f"[ERROR] Kavach CSV not found: {args.kavach_csv}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # -- Build -----------------------------------------------------------------
    merged, stats = build_dataset(args.kavach_csv, args.output_dir)

    # -- Save CSV --------------------------------------------------------------
    csv_out = os.path.join(args.output_dir, "dataset_merged.csv")
    merged.to_csv(csv_out, index=False, encoding="utf-8")
    print(f"[output] Saved merged dataset -> {csv_out}  ({os.path.getsize(csv_out):,} bytes)")

    # -- Save report -----------------------------------------------------------
    report = build_report(merged, stats)
    report_out = os.path.join(args.output_dir, "dataset_report.txt")
    with open(report_out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[output] Saved report        -> {report_out}")

    # -- Print report to stdout ------------------------------------------------
    print()
    print(report)


if __name__ == "__main__":
    main()
