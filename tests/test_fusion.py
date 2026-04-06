"""
tests/test_fusion.py
=====================
Unit tests for kavach.fusion.risk_scorer.RiskScorer.

Coverage:
  1.  Fusion formula weights sum to 1.0 and produce correct blended score
  2.  SAFE level below caution threshold
  3.  CAUTION level at/above caution threshold
  4.  ALERT level at/above alert threshold
  5.  CRITICAL level at/above critical threshold
  6.  Tier accumulation persists across multiple update() calls
  7.  All-3-tiers boost applies when all three tiers accumulated
  8.  All-3-tiers boost is capped at 1.0
  9.  Temporal decay reduces previous score each cycle
  10. instant_escalate bypasses temporal decay (score never decayed down)
  11. reset() clears all state
  12. SLM=None defaults to 0.5 and is handled without error
  13. should_alert=False only for SAFE, True for all others
  14. component_scores dict contains all three keys with correct values
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kavach.detection.heuristics  import HeuristicResult
from kavach.detection.classifier  import ClassifierResult
from kavach.detection.slm         import SLMResult
from kavach.fusion.risk_scorer    import RiskScorer, RiskResult


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — build minimal result objects without touching real models
# ─────────────────────────────────────────────────────────────────────────────

def h(score=0.0, tiers=None, escalate=False):
    return HeuristicResult(
        heuristic_score=score,
        tiers_detected=tiers or [],
        matched_patterns=[],
        instant_escalate=escalate,
    )

def c(p_scam=0.5):
    return ClassifierResult(
        p_scam=p_scam,
        is_suspicious=p_scam >= 0.35,
        escalate_to_slm=p_scam >= 0.35,
        inference_ms=1.0,
    )

def s(verdict="SCAM", tiers=None, p=None, confidence="HIGH"):
    p_map = {"SCAM": 1.0, "LEGITIMATE": 0.0, "UNCERTAIN": 0.5}
    return SLMResult(
        verdict=verdict,
        tiers_detected=tiers or [],
        confidence=confidence,
        reason="test reason",
        p_scam=p if p is not None else p_map[verdict],
        inference_ms=1.0,
    )


@pytest.fixture
def scorer():
    """Fresh RiskScorer with default config each test."""
    return RiskScorer()


@pytest.fixture
def configured_scorer():
    """RiskScorer loaded from config-style dict matching config.yaml."""
    cfg = {
        "fusion": {
            "weight_heuristic":  0.20,
            "weight_classifier": 0.30,
            "weight_slm":        0.50,
            "temporal_decay":    0.90,
        },
        "alerts": {
            "caution":  0.35,
            "alert":    0.65,
            "critical": 0.85,
        },
    }
    return RiskScorer(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Fusion formula weights
# ─────────────────────────────────────────────────────────────────────────────

def test_fusion_formula_weights(configured_scorer):
    """fused = 0.20*h + 0.30*c + 0.50*s, no decay on first call (prev=0)."""
    result = configured_scorer.update(h(0.4), c(0.6), s(p=0.8))
    expected = 0.20 * 0.4 + 0.30 * 0.6 + 0.50 * 0.8  # = 0.08 + 0.18 + 0.40 = 0.66
    assert result.final_score == pytest.approx(expected, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 2–5. Alert level thresholds
# ─────────────────────────────────────────────────────────────────────────────

def test_safe_level(scorer):
    result = scorer.update(h(0.0), c(0.0), s("LEGITIMATE"))
    assert result.alert_level == "SAFE"
    assert not result.should_alert


def test_caution_level(configured_scorer):
    # Score = 0.20*0.15 + 0.30*0.35 + 0.50*0.45 = 0.03+0.105+0.225 = 0.36
    result = configured_scorer.update(h(0.15), c(0.35), s(p=0.45))
    assert result.alert_level == "CAUTION"
    assert result.should_alert


def test_alert_level(configured_scorer):
    # Score = 0.20*0.4 + 0.30*0.6 + 0.50*0.9 = 0.08+0.18+0.45 = 0.71
    result = configured_scorer.update(h(0.4), c(0.6), s(p=0.9))
    assert result.alert_level == "ALERT"
    assert result.should_alert


def test_critical_level(configured_scorer):
    # Score = 0.20*0.5 + 0.30*1.0 + 0.50*1.0 = 0.10+0.30+0.50 = 0.90
    result = configured_scorer.update(h(0.5), c(1.0), s("SCAM"))
    assert result.alert_level == "CRITICAL"
    assert result.should_alert


# ─────────────────────────────────────────────────────────────────────────────
# 6. Tier accumulation persists across update() calls
# ─────────────────────────────────────────────────────────────────────────────

def test_tier_accumulation_persists(scorer):
    scorer.update(h(0.15, tiers=[1]), c(0.3))            # Tier 1 seen
    scorer.update(h(0.10, tiers=[2]), c(0.3))            # Tier 2 seen
    result = scorer.update(h(0.20, tiers=[3]), c(0.3))   # Tier 3 seen

    # All three tiers should be accumulated
    assert 1 in result.tiers_seen_this_call
    assert 2 in result.tiers_seen_this_call
    assert 3 in result.tiers_seen_this_call


def test_tier_1_from_early_call_not_lost(scorer):
    """Tier 1 seen in first window stays even when later windows have no tiers."""
    scorer.update(h(0.15, tiers=[1]), c(0.3))   # Tier 1
    result = scorer.update(h(0.0, tiers=[]),  c(0.2))   # No tiers this window

    assert 1 in result.tiers_seen_this_call   # still accumulated


# ─────────────────────────────────────────────────────────────────────────────
# 7. All-3-tiers boost applies
# ─────────────────────────────────────────────────────────────────────────────

def test_all_three_tiers_boost(configured_scorer):
    """
    Feed all 3 tiers one at a time, then verify the boost is applied
    on the third update.
    """
    configured_scorer.update(h(0.15, tiers=[1]), c(0.3))
    configured_scorer.update(h(0.10, tiers=[2]), c(0.3))

    # Score without boost: 0.20*0.20 + 0.30*0.30 + 0.50*0.5 = 0.04+0.09+0.25=0.38
    # With boost: 0.38 + 0.15 = 0.53 (approximately, after decay of prev)
    result = configured_scorer.update(h(0.20, tiers=[3]), c(0.3))
    # Just confirm the boost was applied (score > pure fusion)
    pure_fused = 0.20 * 0.20 + 0.30 * 0.30 + 0.50 * 0.5
    assert result.final_score > pure_fused


# ─────────────────────────────────────────────────────────────────────────────
# 8. All-3-tiers boost capped at 1.0
# ─────────────────────────────────────────────────────────────────────────────

def test_all_three_tiers_boost_capped(scorer):
    """Score is always <= 1.0 even with boost."""
    scorer.update(h(0.5, tiers=[1, 2]), c(1.0), s("SCAM"))
    result = scorer.update(h(0.5, tiers=[3]), c(1.0), s("SCAM"))
    assert result.final_score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 9. Temporal decay reduces previous score
# ─────────────────────────────────────────────────────────────────────────────

def test_temporal_decay_reduces_score():
    """
    First call: high score. Second call: zero signals.
    Decay should pull the score down (but max(prev, fused) means it stays
    at decayed_prev if fused < decayed_prev).
    """
    scorer = RiskScorer({"fusion": {"temporal_decay": 0.5}})
    r1 = scorer.update(h(0.5), c(1.0), s("SCAM"))    # high score
    r2 = scorer.update(h(0.0), c(0.0), s("LEGITIMATE"))  # zero signals

    # With decay=0.5: prev decayed to r1*0.5, fused=0.5*0=0.  max(prev*0.5, 0.25)
    # Result should be less than r1
    assert r2.final_score < r1.final_score


# ─────────────────────────────────────────────────────────────────────────────
# 10. instant_escalate bypasses temporal decay
# ─────────────────────────────────────────────────────────────────────────────

def test_instant_escalate_locks_score():
    """
    Once instant_escalate fires, score is never decayed down on subsequent
    calls — running score stays as-is before adding the new fused value.
    """
    scorer = RiskScorer({"fusion": {"temporal_decay": 0.1}})  # aggressive decay

    # First call with instant escalate
    r1 = scorer.update(h(0.5, escalate=True), c(1.0), s("SCAM"))
    locked_score = r1.final_score

    # Second call: zero signals, aggressive decay would normally kill the score
    # But instant_escalate locked it — score should stay high
    r2 = scorer.update(h(0.0, escalate=False), c(0.0), s("LEGITIMATE"))

    # Without lock: decayed = locked_score * 0.1 ≈ tiny. With lock: max stays high
    assert r2.final_score >= locked_score * 0.5   # substantially higher than decayed


# ─────────────────────────────────────────────────────────────────────────────
# 11. reset() clears all state
# ─────────────────────────────────────────────────────────────────────────────

def test_reset_clears_state(scorer):
    scorer.update(h(0.5, tiers=[1, 2]), c(1.0), s("SCAM"))
    scorer.reset()

    result = scorer.update(h(0.0), c(0.0), s("LEGITIMATE"))
    assert result.final_score == pytest.approx(0.0)
    assert result.tiers_seen_this_call == []
    assert result.alert_level == "SAFE"


# ─────────────────────────────────────────────────────────────────────────────
# 12. SLM=None defaults cleanly
# ─────────────────────────────────────────────────────────────────────────────

def test_slm_none_no_error(scorer):
    """SLM=None should default s_score=0.5, not crash."""
    result = scorer.update(h(0.15, tiers=[1]), c(0.4), slm=None)
    assert isinstance(result, RiskResult)
    assert result.component_scores["slm"] == pytest.approx(0.5)
    assert result.final_score > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 13. should_alert is False only for SAFE
# ─────────────────────────────────────────────────────────────────────────────

def test_should_alert_false_for_safe(scorer):
    result = scorer.update(h(0.0), c(0.0), s("LEGITIMATE"))
    assert result.alert_level == "SAFE"
    assert result.should_alert is False


def test_should_alert_true_for_caution(configured_scorer):
    result = configured_scorer.update(h(0.15), c(0.35), s(p=0.45))
    assert result.should_alert is True


# ─────────────────────────────────────────────────────────────────────────────
# 14. component_scores contains all three keys
# ─────────────────────────────────────────────────────────────────────────────

def test_component_scores_keys(scorer):
    result = scorer.update(h(0.3), c(0.6), s("SCAM"))
    assert "heuristic" in result.component_scores
    assert "classifier" in result.component_scores
    assert "slm" in result.component_scores
    assert result.component_scores["heuristic"] == pytest.approx(0.3)
    assert result.component_scores["classifier"] == pytest.approx(0.6)
    assert result.component_scores["slm"] == pytest.approx(1.0)
