"""
scripts/bulk_collector.py
==========================
Automated bulk scam-call transcript collector.

Sources:
  1. YouTubeTranscriptApi — fast, free, works for ~40% of English videos
  2. yt-dlp + GeminiASR   — fallback for videos without captions (especially Hindi)

Fixes applied vs original bulk_collecter.py:
  - Fix 1: Removed videoCaption="closedCaption" filter — was silently
            excluding the majority of Hindi/regional videos
  - Fix 2: yt-dlp + GeminiASR fallback for videos without captions
  - Fix 3: MIN_DURATION_S lowered from 30 -> 20 seconds
  - Fix 4: --lang-only flag for per-language runs

Usage:
    python scripts/bulk_collector.py --lang-only en
    python scripts/bulk_collector.py --lang-only hi
    python scripts/bulk_collector.py --lang-only en --use-gemini-fallback False
    python scripts/bulk_collector.py                        # all languages

Config keys used from configs/config.yaml:
    api_keys.youtube   — YouTube Data API v3 key
    api_keys.gemini    — Gemini API key (only needed for --use-gemini-fallback)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ── sys.path so kavach package is importable for GeminiASR ────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

MAX_PER_QUERY  = 20      # videos to check per search query
MIN_DURATION_S = 20      # Fix 3: was 30 — short scam clips (25s OTP request) are valid
MAX_DURATION_S = 1800    # 30 min maximum
MIN_WORDS      = 20      # minimum transcript word count to keep

OUTPUT_FILE    = str(_ROOT / "data" / "raw_transcripts.jsonl")
TMP_DIR        = str(_ROOT / "tmp")

# ─────────────────────────────────────────────────────────────────────────────
# SEARCH QUERIES — 60+ targeted queries across 7 Indian languages
# Each tuple: (query_string, label, language_code)
# label: 1 = scam call,  0 = legitimate banking call
# ─────────────────────────────────────────────────────────────────────────────

SEARCH_QUERIES = [

    # ── HINDI scam calls (hi) ─────────────────────────────────────────────────
    ("KYC scam call recording Hindi",                    1, "hi"),
    ("RBI officer fraud call Hindi recording",           1, "hi"),
    ("OTP scam call Hindi live recording",               1, "hi"),
    ("bijli bill scam call Hindi",                       1, "hi"),
    ("police arrest scam call Hindi recording",          1, "hi"),
    ("SBI bank fraud call Hindi 2024",                   1, "hi"),
    ("paytm KYC scam live call Hindi",                   1, "hi"),
    ("courier FedEx scam call Hindi recording",          1, "hi"),
    ("income tax arrest scam call Hindi",                1, "hi"),
    ("fake bank officer call Hindi recording",           1, "hi"),
    ("loan fraud call recording Hindi",                  1, "hi"),
    ("Jamtara scam call recording Hindi",                1, "hi"),
    ("cyber fraud call Hindi exposed",                   1, "hi"),
    ("ATM block scam call Hindi",                        1, "hi"),
    ("insurance fraud call Hindi recording",             1, "hi"),

    # ── HINDI legitimate banking calls ────────────────────────────────────────
    ("SBI customer care genuine call Hindi",             0, "hi"),
    ("HDFC bank call recording Hindi real",              0, "hi"),
    ("bank EMI reminder call Hindi",                     0, "hi"),
    ("credit card verification call Hindi genuine",      0, "hi"),

    # ── ENGLISH Indian scam calls (en) ────────────────────────────────────────
    ("India call center scam exposed recording",         1, "en"),
    ("KYC update scam call English India",               1, "en"),
    ("Microsoft tech support scam India call",           1, "en"),
    ("Amazon refund scam call India recording",          1, "en"),
    ("Scammer Payback India scam call",                  1, "en"),
    ("Jim Browning India call center scam",              1, "en"),
    ("SSA scam call Indian accent recording",            1, "en"),
    ("bank account suspended scam call India",           1, "en"),
    ("RBI scam call English recording India",            1, "en"),
    ("OTP scam call English India 2024",                 1, "en"),
    ("cyber crime scam call India English",              1, "en"),
    ("Kitboga India scam full call",                     1, "en"),
    ("Pleasant Green India scam call",                   1, "en"),

    # ── ENGLISH legitimate ────────────────────────────────────────────────────
    ("HDFC bank customer service call recording",        0, "en"),
    ("SBI customer care call recording genuine",         0, "en"),
    ("ICICI bank customer care real call",               0, "en"),
    ("bank credit card call legitimate India",           0, "en"),

    # ── TELUGU scam calls (te) ────────────────────────────────────────────────
    ("KYC scam call Telugu recording",                   1, "te"),
    ("bank fraud call Telugu",                           1, "te"),
    ("OTP scam Telugu recording 2024",                   1, "te"),
    ("cyber fraud call Telugu exposed",                  1, "te"),
    ("police scam call Telugu recording",                1, "te"),
    ("loan fraud call Telugu",                           1, "te"),
    ("paytm scam call Telugu",                           1, "te"),
    ("fake bank officer call Telugu",                    1, "te"),

    # ── TELUGU legitimate ─────────────────────────────────────────────────────
    ("bank customer care call Telugu genuine",           0, "te"),
    ("SBI customer service Telugu",                      0, "te"),

    # ── TAMIL scam calls (ta) ─────────────────────────────────────────────────
    ("bank scam call Tamil recording",                   1, "ta"),
    ("KYC fraud call Tamil 2024",                        1, "ta"),
    ("OTP scam Tamil recording",                         1, "ta"),
    ("cyber fraud Tamil call exposed",                   1, "ta"),
    ("police scam call Tamil",                           1, "ta"),
    ("fake RBI call Tamil recording",                    1, "ta"),

    # ── TAMIL legitimate ──────────────────────────────────────────────────────
    ("bank customer care call Tamil",                    0, "ta"),
    ("SBI customer service Tamil recording",             0, "ta"),

    # ── MARATHI scam calls (mr) ───────────────────────────────────────────────
    ("bank fraud call Marathi recording",                1, "mr"),
    ("KYC scam Marathi live call",                       1, "mr"),
    ("cyber crime call Marathi exposed",                 1, "mr"),
    ("OTP fraud Marathi 2024",                           1, "mr"),
    ("police scam call Marathi",                         1, "mr"),

    # ── BENGALI scam calls (bn) ───────────────────────────────────────────────
    ("bank scam call Bangla recording",                  1, "bn"),
    ("KYC fraud Bengali call",                           1, "bn"),
    ("OTP scam Bangla 2024",                             1, "bn"),
    ("cyber fraud call Bengali recording",               1, "bn"),
    ("fake police call Bangla",                          1, "bn"),

    # ── KANNADA scam calls (kn) ───────────────────────────────────────────────
    ("bank fraud call Kannada recording",                1, "kn"),
    ("KYC scam Kannada 2024",                            1, "kn"),
    ("OTP fraud call Kannada",                           1, "kn"),
    ("cyber crime call Kannada exposed",                 1, "kn"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    try:
        import yaml
        cfg_path = _ROOT / "configs" / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: YouTube search — Fix 1: no videoCaption filter
# ─────────────────────────────────────────────────────────────────────────────

# Map query language -> YouTube relevanceLanguage code
_RELEVANCE_LANG = {
    "hi": "hi", "en": "en", "te": "te",
    "ta": "ta", "mr": "mr", "bn": "bn", "kn": "kn",
}


def search_youtube(api_key: str, query: str, lang: str, max_results: int = 20) -> list[str]:
    """
    Search YouTube and return a list of video IDs.

    Fix 1: videoCaption="closedCaption" removed — that filter silently excluded
    most Hindi/regional videos which have auto-captions or none at all.
    We now fetch all videos and try captions + Gemini fallback after.
    """
    try:
        from googleapiclient.discovery import build
        youtube = build("youtube", "v3", developerKey=api_key)

        response = youtube.search().list(
            q               = query,
            part            = "id,snippet",
            type            = "video",
            maxResults      = max_results,
            # Fix 1: videoCaption filter REMOVED
            relevanceLanguage = _RELEVANCE_LANG.get(lang, "en"),
            regionCode      = "IN",
        ).execute()

        return [
            item["id"]["videoId"]
            for item in response.get("items", [])
            if item["id"]["kind"] == "youtube#video"
        ]

    except Exception as e:
        print(f"    [search error] {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Video metadata
# ─────────────────────────────────────────────────────────────────────────────

def get_video_details(api_key: str, video_ids: list[str]) -> dict:
    """
    Batch-fetch video metadata. Returns {video_id: {duration_seconds, title, channel}}.
    Costs 1 quota unit per video.
    """
    if not video_ids:
        return {}
    try:
        from googleapiclient.discovery import build
        import isodate

        youtube  = build("youtube", "v3", developerKey=api_key)
        response = youtube.videos().list(
            id   = ",".join(video_ids),
            part = "contentDetails,snippet",
        ).execute()

        details = {}
        for item in response.get("items", []):
            vid_id   = item["id"]
            duration = item["contentDetails"]["duration"]   # ISO 8601: PT4M33S
            try:
                secs = int(isodate.parse_duration(duration).total_seconds())
            except Exception:
                m = re.search(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration)
                h, mn, s = (int(x) if x else 0 for x in m.groups()) if m else (0, 0, 0)
                secs = h * 3600 + mn * 60 + s

            details[vid_id] = {
                "duration_seconds": secs,
                "title":   item["snippet"]["title"],
                "channel": item["snippet"]["channelTitle"],
            }
        return details

    except ImportError:
        print("    [warn] isodate not installed: pip install isodate")
        return {}
    except Exception as e:
        print(f"    [details error] {e}")
        return {}


def passes_duration_filter(duration_s: int) -> bool:
    return MIN_DURATION_S <= duration_s <= MAX_DURATION_S


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3a: Primary transcript — YouTubeTranscriptApi
# ─────────────────────────────────────────────────────────────────────────────

_LANG_PRIORITY = {
    "hi": ["hi", "hi-IN", "en", "en-IN"],
    "en": ["en", "en-IN", "en-GB", "en-US"],
    "te": ["te", "en"],
    "ta": ["ta", "en"],
    "mr": ["mr", "hi", "en"],
    "bn": ["bn", "en"],
    "kn": ["kn", "en"],
}

# Singleton API instance — youtube-transcript-api v1.x uses instance-based API
_yt_api_instance = None


def _get_yt_api():
    global _yt_api_instance
    if _yt_api_instance is None:
        from youtube_transcript_api import YouTubeTranscriptApi
        _yt_api_instance = YouTubeTranscriptApi()
    return _yt_api_instance


def _snippets_to_text_and_segs(fetched) -> tuple[str, list]:
    """Convert FetchedTranscript (v1.x) to (full_text, segments_list)."""
    snippets = fetched.snippets
    text = " ".join(s.text for s in snippets)
    segs = [{"text": s.text, "start": s.start, "duration": s.duration}
            for s in snippets]
    return text, segs


def fetch_transcript_api(video_id: str, target_lang: str) -> tuple[str | None, list | None, str | None]:
    """
    Try YouTubeTranscriptApi (v1.x instance API) for the video.
    Returns (text, segments, detected_lang) or (None, None, None).

    v1.x change: class-level get_transcript/list_transcripts removed.
    Now uses api.fetch() and api.list() on an instance.
    fetch() returns FetchedTranscript with .snippets (not a list of dicts).
    """
    try:
        api        = _get_yt_api()
        lang_order = _LANG_PRIORITY.get(target_lang, ["en"])

        # Try each preferred language directly
        for lang in lang_order:
            try:
                fetched = api.fetch(video_id, languages=[lang])
                text, segs = _snippets_to_text_and_segs(fetched)
                return text, segs, lang
            except Exception:
                continue

        # Last resort: inspect all available transcripts
        try:
            tlist = api.list(video_id)
            try:
                t = tlist.find_manually_created_transcript(lang_order)
            except Exception:
                t = tlist.find_generated_transcript(lang_order)
            fetched = t.fetch()
            text, segs = _snippets_to_text_and_segs(fetched)
            return text, segs, t.language_code
        except Exception:
            return None, None, None

    except Exception:
        return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3b: Fallback — yt-dlp download + GeminiASR transcription  (Fix 2)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_transcript_gemini(
    video_id: str,
    gemini_api_key: str,
    tmp_dir: str = TMP_DIR,
) -> tuple[str | None, str]:
    """
    Fix 2: Fallback for videos without captions.
    Downloads audio via yt-dlp, transcribes with GeminiASR.

    Returns (text, method_tag) where method_tag is "gemini" on success
    or an error string starting with "error:" on failure.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    video_url   = f"https://www.youtube.com/watch?v={video_id}"
    out_template = os.path.join(tmp_dir, f"%(id)s.%(ext)s")

    # ── Download audio with yt-dlp ────────────────────────────────────────────
    try:
        result = subprocess.run(
            [
                "yt-dlp", "-x",
                "--audio-format", "mp3",
                "--audio-quality", "5",
                "--no-playlist",
                "-o", out_template,
                video_url,
            ],
            capture_output = True,
            text           = True,
            timeout        = 180,
        )
        if result.returncode != 0:
            err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown"
            return None, f"error:yt-dlp({err[:80]})"
    except FileNotFoundError:
        return None, "error:yt-dlp not installed (pip install yt-dlp)"
    except subprocess.TimeoutExpired:
        return None, "error:yt-dlp timeout"
    except Exception as e:
        return None, f"error:yt-dlp({e})"

    # ── Find downloaded file ──────────────────────────────────────────────────
    mp3_path = os.path.join(tmp_dir, f"{video_id}.mp3")
    if not os.path.exists(mp3_path):
        # yt-dlp sometimes produces webm/m4a that ffmpeg then converts
        for fname in os.listdir(tmp_dir):
            if video_id in fname:
                mp3_path = os.path.join(tmp_dir, fname)
                break
        else:
            return None, "error:downloaded file not found"

    # ── Transcribe with GeminiASR ─────────────────────────────────────────────
    try:
        import librosa
        import numpy as np
        audio, _ = librosa.load(mp3_path, sr=16_000, mono=True)
        audio = audio.astype(np.float32)
    except Exception as e:
        return None, f"error:librosa({e})"
    finally:
        try:
            os.remove(mp3_path)
        except Exception:
            pass

    try:
        from kavach.transcription.gemini_asr import GeminiASR
        # Use 1.5-flash for bulk collection: 1,500 req/day free tier vs 20 for 2.5-flash
        asr  = GeminiASR(api_key=gemini_api_key, model_name="gemini-1.5-flash")
        text = asr.transcribe_raw(audio)
        if not text or not text.strip():
            return None, "error:gemini returned empty transcript"
        return text.strip(), "gemini"
    except Exception as e:
        return None, f"error:GeminiASR({e})"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Quality check
# ─────────────────────────────────────────────────────────────────────────────

def quality_check(text: str, label: int) -> tuple[bool, str]:
    """Returns (passes, reason)."""
    wc = len(text.split())
    if wc < MIN_WORDS:
        return False, f"too short ({wc} words)"
    if wc > 15_000:
        return False, f"too long ({wc} words) — likely a documentary"

    text_lower = text.lower()

    call_markers = [
        "hello", "hi ", "calling", "speaking", "yes ", "okay",
        "please", "account", "bank", "haan", "namaste", "allo",
    ]
    if sum(1 for m in call_markers if m in text_lower) < 2:
        return False, "does not look like a phone call transcript"

    if label == 1:
        scam_signals = [
            "otp", "pin", "account number", "aadhaar", "arrest",
            "suspended", "frozen", "transfer", "kyc", "verify",
            "password", "cvv", "debit", "atm", "otp bhejo",
            "account band", "police", "courier",
        ]
        if sum(1 for s in scam_signals if s in text_lower) == 0:
            return False, "no scam signals found — may be misclassified"

    return True, "ok"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN COLLECTION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_collection(
    yt_api_key:         str,
    gemini_api_key:     str,
    use_gemini_fallback: bool,
    lang_only:          str | None,
    output_file:        str,
    max_per_query:      int,
) -> list[dict]:

    # ── Optionally filter to one language (Fix 4) ─────────────────────────────
    queries = SEARCH_QUERIES
    if lang_only:
        queries = [(q, l, lg) for q, l, lg in SEARCH_QUERIES if lg == lang_only]
        if not queries:
            print(f"[ERROR] No queries found for --lang-only '{lang_only}'")
            print(f"        Valid codes: {sorted(set(lg for _,_,lg in SEARCH_QUERIES))}")
            sys.exit(1)
        print(f"[lang-only] Running {len(queries)} queries for language='{lang_only}'")

    # ── Resume support ────────────────────────────────────────────────────────
    seen_video_ids: set[str] = set()
    records: list[dict]     = []

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(output_file):
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    seen_video_ids.add(r["video_id"])
                    records.append(r)
                except Exception:
                    pass
        print(f"[resume] {len(seen_video_ids)} videos already collected")

    stats = defaultdict(lambda: {"searched": 0, "api_ok": 0, "gemini_ok": 0,
                                  "no_transcript": 0, "failed_qc": 0})

    output_handle = open(output_file, "a", encoding="utf-8")

    total_queries = len(queries)
    for q_idx, (query, label, lang) in enumerate(queries, 1):
        print(f"\n[{q_idx}/{total_queries}] '{query}'  lang={lang}  label={label}")

        # Search
        video_ids = search_youtube(yt_api_key, query, lang, max_results=max_per_query)
        new_ids   = [v for v in video_ids if v not in seen_video_ids]
        print(f"  Found {len(video_ids)} videos, {len(new_ids)} new")
        stats[lang]["searched"] += len(new_ids)

        if not new_ids:
            time.sleep(0.5)
            continue

        # Fetch metadata for duration filter
        details = get_video_details(yt_api_key, new_ids)

        for vid_id in new_ids:
            info = details.get(vid_id, {})
            dur  = info.get("duration_seconds", 0)

            if not passes_duration_filter(dur):
                print(f"    {vid_id}: skip duration={dur}s")
                continue

            video_url = f"https://www.youtube.com/watch?v={vid_id}"

            # ── Primary: YouTubeTranscriptApi ──────────────────────────────
            text, segments, detected_lang = fetch_transcript_api(vid_id, lang)
            transcript_method = "api"

            # ── Fallback: yt-dlp + GeminiASR (Fix 2) ──────────────────────
            if text is None and use_gemini_fallback:
                print(f"    {vid_id}: no caption — trying Gemini fallback ...")
                text, method_tag = fetch_transcript_gemini(vid_id, gemini_api_key)
                if text:
                    transcript_method = "gemini"
                    detected_lang     = lang     # accept query language as detected
                    segments          = []       # no timestamped segments from Gemini
                else:
                    print(f"    {vid_id}: {method_tag}")

            if text is None:
                print(f"    {vid_id}: no transcript available")
                stats[lang]["no_transcript"] += 1
                continue

            # ── Quality check ──────────────────────────────────────────────
            passes, reason = quality_check(text, label)
            if not passes:
                print(f"    {vid_id}: FAIL QC — {reason}")
                stats[lang]["failed_qc"] += 1
                continue

            # ── Build and save record ──────────────────────────────────────
            record = {
                "video_id":           vid_id,
                "url":                video_url,
                "title":              info.get("title", ""),
                "channel":            info.get("channel", ""),
                "duration_seconds":   dur,
                "full_text":          text,
                "segments":           segments,
                "label":              label,
                "label_name":         "scam" if label == 1 else "legitimate",
                "language":           lang,
                "detected_lang":      detected_lang,
                "transcript_method":  transcript_method,
                "source":             "youtube",
                "search_query":       query,
                "word_count":         len(text.split()),
                "conversation_id":    f"yt_{vid_id}",
                "collected_at":       datetime.now().isoformat(),
            }

            seen_video_ids.add(vid_id)
            records.append(record)
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_handle.flush()

            key = "gemini_ok" if transcript_method == "gemini" else "api_ok"
            stats[lang][key] += 1
            print(
                f"    {vid_id}: OK  words={record['word_count']}"
                f"  lang={detected_lang}  method={transcript_method}"
            )

            time.sleep(0.3)

        time.sleep(1.5)   # be polite between queries

    output_handle.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  COLLECTION COMPLETE -- {len(records)} total records")
    print(f"{'='*60}")

    scam_count  = sum(1 for r in records if r["label"] == 1)
    legit_count = sum(1 for r in records if r["label"] == 0)
    api_count   = sum(1 for r in records if r.get("transcript_method") == "api")
    gem_count   = sum(1 for r in records if r.get("transcript_method") == "gemini")

    print(f"  Scam      : {scam_count}")
    print(f"  Legitimate: {legit_count}")
    print(f"  Via API   : {api_count}")
    print(f"  Via Gemini: {gem_count}")
    print(f"\n  Per language:")

    lang_counts: dict[str, int] = defaultdict(int)
    for r in records:
        lang_counts[r["language"]] += 1
    for lg, cnt in sorted(lang_counts.items(), key=lambda x: -x[1]):
        s = stats[lg]
        print(
            f"    {lg}: {cnt} kept  "
            f"(searched={s['searched']}  api={s['api_ok']}  "
            f"gemini={s['gemini_ok']}  no_transcript={s['no_transcript']}  "
            f"failed_qc={s['failed_qc']})"
        )

    print(f"\n  Saved to: {output_file}")
    print(f"  Next: python scripts/build_dataset.py")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk YouTube scam-call transcript collector."
    )
    parser.add_argument(
        "--lang-only",
        metavar="LANG",
        default=None,
        help="Only run queries for this language code (e.g. en, hi, te, ta, mr, bn, kn)",
    )
    parser.add_argument(
        "--use-gemini-fallback",
        type=lambda x: x.lower() not in ("false", "0", "no"),
        default=True,
        metavar="BOOL",
        help="Fall back to yt-dlp + GeminiASR when no caption is available (default: True)",
    )
    parser.add_argument(
        "--max-per-query",
        type=int,
        default=MAX_PER_QUERY,
        help=f"Videos to check per search query (default: {MAX_PER_QUERY})",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_FILE,
        help=f"Output JSONL file (default: {OUTPUT_FILE})",
    )
    parser.add_argument(
        "--yt-key",
        default=None,
        help="YouTube Data API v3 key (overrides config.yaml)",
    )
    parser.add_argument(
        "--gemini-key",
        default=None,
        help="Gemini API key (overrides config.yaml, only needed with --use-gemini-fallback)",
    )
    args = parser.parse_args()

    # ── Resolve API keys: CLI > config.yaml ───────────────────────────────────
    cfg = _load_config()
    api_keys = cfg.get("api_keys", {})

    yt_key     = args.yt_key     or api_keys.get("youtube", "")
    gemini_key = args.gemini_key or api_keys.get("gemini",  "")

    if not yt_key:
        print("[ERROR] No YouTube API key found.")
        print("        Set api_keys.youtube in configs/config.yaml or use --yt-key KEY")
        sys.exit(1)

    if args.use_gemini_fallback and not gemini_key:
        print("[WARN] --use-gemini-fallback is True but no Gemini key found.")
        print("       Set api_keys.gemini in configs/config.yaml or use --gemini-key KEY")
        print("       Continuing with caption-only mode.")
        args.use_gemini_fallback = False

    print(f"[config] YouTube API key : {'*' * (len(yt_key) - 4) + yt_key[-4:]}")
    print(f"[config] Gemini fallback : {args.use_gemini_fallback}")
    print(f"[config] Lang filter     : {args.lang_only or 'all'}")
    print(f"[config] Max per query   : {args.max_per_query}")
    print(f"[config] Output          : {args.output}")
    print(f"[config] MIN_DURATION_S  : {MIN_DURATION_S}s  (Fix 3: was 30s)")
    print()

    run_collection(
        yt_api_key          = yt_key,
        gemini_api_key      = gemini_key,
        use_gemini_fallback = args.use_gemini_fallback,
        lang_only           = args.lang_only,
        output_file         = args.output,
        max_per_query       = args.max_per_query,
    )


if __name__ == "__main__":
    main()
