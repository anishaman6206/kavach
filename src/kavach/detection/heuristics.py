"""
kavach.detection.heuristics
=============================
Regex-based Manipulation Funnel tier detector.

Runs on buf.caller_text_only() — CALLER speech only, already lowercased.
Never makes a final scam/safe decision; contributes heuristic_score (0.0–0.5)
to the downstream risk fusion layer.

The only exception is instant_escalate: ~10 catastrophic phrases that no
legitimate bank, courier, or government body ever utters. When any of these
match, the result is escalated immediately without waiting for MuRIL/SLM.

Design decisions:
  - All patterns anchored with word boundaries (\b) to avoid substring false
    positives (e.g. "bank" matching "bankruptcy").
  - Patterns cover both English and transliterated Hindi (Latin script) because
    that's what Whisper / IndicConformer output for mixed-language speech.
  - Devanagari patterns deliberately excluded here — Devanagari text goes to
    MuRIL (which handles it natively). Heuristics stay in ASCII/Latin.
  - Scoring follows the Manipulation Funnel logic: Tier 1 + Tier 3 together
    is the minimum required for a banking scam, so their co-presence gets a
    strong combined boost.

Scoring table:
  Tier 1 only              → 0.15
  Tier 2 only              → 0.10
  Tier 3 only              → 0.20
  Tier 1 + Tier 2          → 0.30
  Tier 1 + Tier 3          → 0.40
  Tier 2 + Tier 3          → 0.30
  All three tiers          → 0.50
  instant_escalate = True  → 0.50 (regardless of tier matches)

Usage:
    from kavach.detection.heuristics import HeuristicDetector

    detector = HeuristicDetector()
    result = detector.analyze(buf.caller_text_only())
    print(result.heuristic_score, result.tiers_detected)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Pattern, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HeuristicResult:
    """
    Output of HeuristicDetector.analyze().

    Attributes:
        heuristic_score  : float 0.0–0.5, fed into risk_scorer fusion layer.
        tiers_detected   : list of Manipulation Funnel tiers present (1, 2, 3).
        matched_patterns : human-readable list of which pattern labels fired.
        instant_escalate : True if a catastrophic phrase was matched —
                           bypass ML and escalate immediately.
    """
    heuristic_score  : float
    tiers_detected   : List[int]
    matched_patterns : List[str]
    instant_escalate : bool

    def __repr__(self) -> str:
        return (
            f"HeuristicResult("
            f"score={self.heuristic_score:.2f}, "
            f"tiers={self.tiers_detected}, "
            f"escalate={self.instant_escalate}, "
            f"patterns={self.matched_patterns})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pattern definitions
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (label, regex_pattern)
# Patterns use raw strings; \b for word boundaries where appropriate.

_TIER1_PATTERNS: List[Tuple[str, str]] = [
    # ── Indian banks & financial regulators ─────────────────────────────────
    ("t1_rbi",            r"\brbi\b"),
    ("t1_rbi_officer",    r"\brbi\s+officer\b"),
    ("t1_rbi_hindi",      r"\brbi\s+ki\s+taraf\s+se\b"),
    ("t1_sbi",            r"\bsbi\b"),
    ("t1_sbi_hindi",      r"\bsbi\s+se\s+bol\s+rah"),
    ("t1_icici",          r"\bicici\b"),
    ("t1_hdfc",           r"\bhdfc\b"),
    ("t1_axis_bank",      r"\baxis\s+bank\b"),
    ("t1_bank_officer",   r"\bbank\s+officer\b"),
    ("t1_bank_manager",   r"\bbank\s+manager\b"),
    ("t1_calling_from_bank", r"\bcalling\s+from\s+\w*\s*bank\b"),
    # ── Government / law enforcement ────────────────────────────────────────
    ("t1_police",         r"\bpolice\b"),
    ("t1_police_hindi",   r"\bpolice\s+station\s+se\b"),
    ("t1_cbi",            r"\bcbi\b"),
    ("t1_income_tax",     r"\bincome\s+tax\b"),
    ("t1_cybercrime",     r"\bcyber\s*crime\b"),
    ("t1_government",     r"\bgovernment\s+officer\b"),
    ("t1_court",          r"\bcourt\s+notice\b"),
    # ── E-commerce / courier ────────────────────────────────────────────────
    ("t1_amazon",         r"\bamazon\b"),
    ("t1_flipkart",       r"\bflipkart\b"),
    ("t1_fedex",          r"\bfedex\b"),
    ("t1_dhl",            r"\bdhl\b"),
    ("t1_courier",        r"\bcourier\b"),
    ("t1_delivery_agent", r"\bdelivery\s+agent\b"),
    # ── Telecom / TRAI ───────────────────────────────────────────────────────
    ("t1_trai",           r"\btrai\b"),
    ("t1_telecom",        r"\btelecom\s+department\b"),
]

_TIER2_PATTERNS: List[Tuple[str, str]] = [
    # ── Account threats ──────────────────────────────────────────────────────
    ("t2_account_blocked",    r"\baccount\s+(blocked|suspended|frozen|closed|deactivated)\b"),
    ("t2_account_band",       r"\baccount\s+band\s+ho"),
    ("t2_account_freeze",     r"\baccount\s+will\s+be\s+freeze\b"),
    # ── Arrest / legal threats ───────────────────────────────────────────────
    ("t2_arrested",           r"\b(arrested|arrest\s+warrant)\b"),
    ("t2_giraftaar",          r"\bgiraftaar\b"),
    ("t2_legal_action",       r"\blegal\s+action\b"),
    ("t2_case_filed",         r"\bcase\s+(filed|registered)\b"),
    ("t2_fir",                r"\bfir\s+(darj|filed|register)\b"),
    # ── Time pressure ────────────────────────────────────────────────────────
    ("t2_last_chance",        r"\blast\s+chance\b"),
    ("t2_immediately",        r"\bimmediately\b"),
    ("t2_right_now",          r"\bright\s+now\b"),
    ("t2_abhi_karo",          r"\babhi\s+karo\b"),
    ("t2_turant",             r"\bturant\b"),
    ("t2_2_hours",            r"\b2\s+hours?\b"),
    ("t2_30_minutes",         r"\b30\s+minutes?\b"),
    ("t2_time_limit",         r"\bwithin\s+\d+\s+(hour|minute|min)\b"),
    # ── Urgency phrases ──────────────────────────────────────────────────────
    ("t2_urgent",             r"\burgent(ly)?\b"),
    ("t2_emergency",          r"\bemergency\b"),
    ("t2_dont_tell",          r"\bdon.?t\s+tell\s+anyone\b"),
    ("t2_keep_secret",        r"\bkeep\s+(it\s+)?secret\b"),
]

_TIER3_PATTERNS: List[Tuple[str, str]] = [
    # ── OTP / PIN ────────────────────────────────────────────────────────────
    ("t3_otp",                r"\botp\b"),
    ("t3_otp_batao",          r"\botp\s+batao\b"),
    ("t3_otp_share",          r"\botp\s+share\b"),
    ("t3_otp_send",           r"\botp\s+(send|bhejo|de\s+do|dena)\b"),
    ("t3_pin",                r"\b(atm\s+)?pin\b"),
    ("t3_cvv",                r"\bcvv\b"),
    # ── Account / card credentials ───────────────────────────────────────────
    ("t3_account_number",     r"\baccount\s+number\b"),
    ("t3_card_number",        r"\bcard\s+number\b"),
    ("t3_card_details",       r"\bcard\s+details\b"),
    ("t3_expiry",             r"\bexpiry\s+(date|number)\b"),
    # ── Identity documents ────────────────────────────────────────────────────
    ("t3_aadhaar",            r"\baadhaar\b"),
    ("t3_pan_card",           r"\bpan\s+card\b"),
    ("t3_pan_number",         r"\bpan\s+number\b"),
    # ── Generic extraction ────────────────────────────────────────────────────
    ("t3_password",           r"\bpassword\b"),
    ("t3_share_your",         r"\bshare\s+your\b"),
    ("t3_tell_me_your",       r"\btell\s+me\s+your\b"),
    ("t3_number_batao",       r"\bnumber\s+batao\b"),
    # ── WhatsApp / remote extraction ─────────────────────────────────────────
    ("t3_whatsapp",           r"\bwhatsapp\b"),
    ("t3_whatsapp_bhejo",     r"\bwhatsapp\s+pe\s+bhejo\b"),
    ("t3_send_on_whatsapp",   r"\bsend\s+(it\s+)?on\s+whatsapp\b"),
    ("t3_screenshot",         r"\bscreenshot\s+(le|lo|send|bhejo)\b"),
    # ── Money transfer ────────────────────────────────────────────────────────
    ("t3_transfer_money",     r"\btransfer\s+(the\s+)?money\b"),
    ("t3_paisa_bhejo",        r"\bpaisa\s+bhejo\b"),
]

# Catastrophic phrases — no legitimate entity ever says these.
# instant_escalate = True when any of these match.
_INSTANT_ESCALATE_PATTERNS: List[Tuple[str, str]] = [
    ("esc_rbi_safe_account",     r"\btransfer\s+to\s+(rbi\s+)?safe\s+account\b"),
    ("esc_aadhaar_drug",         r"\baadhaar\s+(linked|connected)\s+to\s+(drug|narcotics|crime)\b"),
    ("esc_arrest_warrant",       r"\barrest\s+warrant\s+(issued|generated|against\s+you)\b"),
    ("esc_account_frozen_in",    r"\baccount\s+will\s+be\s+frozen\s+in\b"),
    ("esc_share_otp_or_account", r"\bshare\s+otp\s+or\s+(your\s+)?account\b"),
    ("esc_send_money_to_verify", r"\bsend\s+money\s+to\s+verify\b"),
    ("esc_rbi_digital_arrest",   r"\bdigital\s+arrest\b"),
    ("esc_custom_duty_payment",  r"\bcustom\s+duty\s+payment\b"),
    ("esc_drug_parcel",          r"\bdrug\s+parcel\b"),
    ("esc_money_laundering",     r"\bmoney\s+laundering\s+case\b"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Score lookup
# ─────────────────────────────────────────────────────────────────────────────

def _compute_score(tiers: List[int], escalate: bool) -> float:
    if escalate:
        return 0.50
    s = frozenset(tiers)
    if s == {1, 2, 3}:
        return 0.50
    if s == {1, 3}:
        return 0.40
    if s == {1, 2}:
        return 0.30
    if s == {2, 3}:
        return 0.30
    if s == {3}:
        return 0.20
    if s == {1}:
        return 0.15
    if s == {2}:
        return 0.10
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────

class HeuristicDetector:
    """
    Regex-based Manipulation Funnel tier detector.

    Compile-once on init; analyze() is O(n_patterns) per call — fast enough
    to run after every Whisper/GeminiASR utterance flush.

    Args:
        extra_tier1 : additional (label, pattern) pairs to inject into Tier 1.
        extra_tier2 : same for Tier 2.
        extra_tier3 : same for Tier 3.
        extra_escalate : same for instant-escalate patterns.

    Usage:
        detector = HeuristicDetector()
        result = detector.analyze("sbi se bol raha hoon otp batao")
        # HeuristicResult(score=0.40, tiers=[1, 3], escalate=False, ...)
    """

    def __init__(
        self,
        extra_tier1: List[Tuple[str, str]] | None = None,
        extra_tier2: List[Tuple[str, str]] | None = None,
        extra_tier3: List[Tuple[str, str]] | None = None,
        extra_escalate: List[Tuple[str, str]] | None = None,
    ) -> None:
        self._tier1    = self._compile(_TIER1_PATTERNS    + (extra_tier1    or []))
        self._tier2    = self._compile(_TIER2_PATTERNS    + (extra_tier2    or []))
        self._tier3    = self._compile(_TIER3_PATTERNS    + (extra_tier3    or []))
        self._escalate = self._compile(_INSTANT_ESCALATE_PATTERNS + (extra_escalate or []))

    @staticmethod
    def _compile(patterns: List[Tuple[str, str]]) -> List[Tuple[str, Pattern]]:
        compiled = []
        for label, pat in patterns:
            compiled.append((label, re.compile(pat, re.IGNORECASE)))
        return compiled

    @staticmethod
    def _scan(text: str, compiled: List[Tuple[str, Pattern]]) -> List[str]:
        """Return list of labels whose pattern matched in text."""
        return [label for label, rx in compiled if rx.search(text)]

    def analyze(self, caller_text: str) -> HeuristicResult:
        """
        Analyze CALLER-only text for Manipulation Funnel tier signals.

        Args:
            caller_text : output of buf.caller_text_only() — lowercased plain string.

        Returns:
            HeuristicResult with score, tiers, matched patterns, escalate flag.
        """
        text = caller_text.lower()

        escalate_hits = self._scan(text, self._escalate)
        t1_hits = self._scan(text, self._tier1)
        t2_hits = self._scan(text, self._tier2)
        t3_hits = self._scan(text, self._tier3)

        tiers_detected: List[int] = []
        if t1_hits:
            tiers_detected.append(1)
        if t2_hits:
            tiers_detected.append(2)
        if t3_hits:
            tiers_detected.append(3)

        instant_escalate = bool(escalate_hits)
        matched_patterns = escalate_hits + t1_hits + t2_hits + t3_hits
        score = _compute_score(tiers_detected, instant_escalate)

        return HeuristicResult(
            heuristic_score=score,
            tiers_detected=tiers_detected,
            matched_patterns=matched_patterns,
            instant_escalate=instant_escalate,
        )
