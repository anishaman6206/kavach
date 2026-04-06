"""
tests/test_heuristics.py
=========================
Unit tests for kavach.detection.heuristics.HeuristicDetector.

Coverage:
  - English scam patterns (each tier individually + combined)
  - Hindi / transliterated scam patterns
  - Legitimate bank call (should not flag)
  - Netflix OTP (Tier 3 only — low score, no escalate)
  - Instant-escalate catastrophic phrases
  - Mixed-language (Hindi + English) input
  - Score correctness for all tier combinations
  - Empty input edge case
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kavach.detection.heuristics import HeuristicDetector, HeuristicResult


@pytest.fixture(scope="module")
def det() -> HeuristicDetector:
    return HeuristicDetector()


# ── 1. English Tier 1 only ────────────────────────────────────────────────────

def test_english_tier1_only(det):
    result = det.analyze("i am calling from sbi customer care")
    assert 1 in result.tiers_detected
    assert result.heuristic_score == pytest.approx(0.15)
    assert not result.instant_escalate


# ── 2. English Tier 3 only ────────────────────────────────────────────────────

def test_english_tier3_only(det):
    result = det.analyze("please share your otp with me")
    assert 3 in result.tiers_detected
    assert 1 not in result.tiers_detected
    assert result.heuristic_score == pytest.approx(0.20)
    assert not result.instant_escalate


# ── 3. English Tier 2 only ───────────────────────────────────────────────────

def test_english_tier2_only(det):
    result = det.analyze("your account has been blocked immediately")
    assert 2 in result.tiers_detected
    assert result.heuristic_score == pytest.approx(0.10)
    assert not result.instant_escalate


# ── 4. Tier 1 + Tier 3 — classic banking scam ────────────────────────────────

def test_tier1_plus_tier3(det):
    result = det.analyze("calling from icici bank please tell me your cvv and otp")
    assert 1 in result.tiers_detected
    assert 3 in result.tiers_detected
    assert result.heuristic_score == pytest.approx(0.40)
    assert not result.instant_escalate


# ── 5. All three tiers ────────────────────────────────────────────────────────

def test_all_three_tiers(det):
    # Use Tier 2 phrase that is NOT an instant-escalate phrase
    result = det.analyze(
        "i am rbi officer legal action will be taken against you immediately "
        "share your otp right now"
    )
    assert result.tiers_detected == [1, 2, 3]
    assert result.heuristic_score == pytest.approx(0.50)
    assert not result.instant_escalate


# ── 6. Hindi Tier 1 — transliterated ─────────────────────────────────────────

def test_hindi_tier1(det):
    result = det.analyze("sbi se bol raha hoon aapko ek important call hai")
    assert 1 in result.tiers_detected
    assert result.heuristic_score >= 0.15


# ── 7. Hindi Tier 3 — OTP extraction ─────────────────────────────────────────

def test_hindi_tier3_otp(det):
    result = det.analyze("aapka otp batao jaldi karo")
    assert 3 in result.tiers_detected
    assert result.heuristic_score >= 0.20


# ── 8. Hindi mixed scam — Tier 1 + Tier 3 ────────────────────────────────────

def test_hindi_mixed_scam(det):
    # "account band ho jayega" is Tier 2 — use input with only Tier 1 + Tier 3
    result = det.analyze(
        "rbi ki taraf se call kar raha hoon "
        "otp share karo please"
    )
    assert 1 in result.tiers_detected
    assert 3 in result.tiers_detected
    assert 2 not in result.tiers_detected
    assert result.heuristic_score == pytest.approx(0.40)


# ── 9. Legitimate bank call — should not flag ─────────────────────────────────

def test_legit_bank_call_no_flag(det):
    """Real bank calls to confirm existing transactions never ask for OTP."""
    result = det.analyze(
        "hello this is a courtesy call from your bank "
        "we wanted to inform you that your statement is ready "
        "please visit the branch for any queries thank you"
    )
    assert result.heuristic_score == pytest.approx(0.0)
    assert result.tiers_detected == []
    assert not result.instant_escalate


# ── 10. Netflix OTP — Tier 3 only, low score ─────────────────────────────────

def test_netflix_otp_tier3_only(det):
    """
    'bro share otp for netflix' must NOT trigger Tier 1 or escalate.
    This is the exact false-positive the SLM layer solves — heuristics
    correctly score it low (0.20, Tier 3 only) and leave the decision to SLM.
    """
    result = det.analyze("bro share your otp for netflix subscription")
    assert 3 in result.tiers_detected
    assert 1 not in result.tiers_detected
    assert result.heuristic_score == pytest.approx(0.20)
    assert not result.instant_escalate


# ── 11. Instant escalate — RBI safe account ──────────────────────────────────

def test_instant_escalate_rbi_safe_account(det):
    result = det.analyze("you need to transfer to rbi safe account immediately")
    assert result.instant_escalate is True
    assert result.heuristic_score == pytest.approx(0.50)


# ── 12. Instant escalate — arrest warrant ────────────────────────────────────

def test_instant_escalate_arrest_warrant(det):
    result = det.analyze("arrest warrant issued against your name by cybercrime")
    assert result.instant_escalate is True
    assert result.heuristic_score == pytest.approx(0.50)
    assert "esc_arrest_warrant" in result.matched_patterns


# ── 13. Instant escalate — digital arrest ────────────────────────────────────

def test_instant_escalate_digital_arrest(det):
    result = det.analyze("you are under digital arrest do not leave your house")
    assert result.instant_escalate is True
    assert result.heuristic_score == pytest.approx(0.50)


# ── 14. Mixed Hindi + English — real-world code-switching ────────────────────

def test_mixed_language_amazon_icici(det):
    """
    Mirrors the actual Gemini transcript from our scam_call.mp3:
    Amazon ICICI credit card, WhatsApp verification, OTP extraction.
    """
    result = det.analyze(
        "amazon icici ka jo platinum chip card milega aapko "
        "no annual fee no joining fee whatsapp pe verification karaata hoon "
        "otp batao sir"
    )
    assert 1 in result.tiers_detected   # amazon + icici
    assert 3 in result.tiers_detected   # otp + whatsapp
    assert result.heuristic_score == pytest.approx(0.40)
    assert not result.instant_escalate


# ── 15. Drug parcel instant escalate ─────────────────────────────────────────

def test_instant_escalate_drug_parcel(det):
    result = det.analyze(
        "a drug parcel was found registered under your aadhaar number"
    )
    assert result.instant_escalate is True
    assert result.heuristic_score == pytest.approx(0.50)


# ── 16. Tier 1 + Tier 2 combo ────────────────────────────────────────────────

def test_tier1_tier2_combo(det):
    result = det.analyze(
        "i am calling from cybercrime department legal action will be taken immediately"
    )
    assert 1 in result.tiers_detected
    assert 2 in result.tiers_detected
    assert 3 not in result.tiers_detected
    assert result.heuristic_score == pytest.approx(0.30)


# ── 17. Empty input ───────────────────────────────────────────────────────────

def test_empty_input(det):
    result = det.analyze("")
    assert result.heuristic_score == pytest.approx(0.0)
    assert result.tiers_detected == []
    assert result.matched_patterns == []
    assert not result.instant_escalate


# ── 18. matched_patterns is populated correctly ───────────────────────────────

def test_matched_patterns_populated(det):
    result = det.analyze("calling from sbi bank otp share karo")
    assert len(result.matched_patterns) > 0
    # At minimum sbi (t1) and otp (t3) should be in matched patterns
    labels = result.matched_patterns
    assert any("t1_sbi" in l for l in labels)
    assert any("t3_otp" in l for l in labels)
