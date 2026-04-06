"""
kavach.fusion.risk_scorer
==========================
Weighted score fusion + tier accumulation + temporal decay.

Sits at the end of the detection pipeline, consuming all three signal
outputs and producing a single authoritative RiskResult per call cycle.

Pipeline position:
    HeuristicDetector ─┐
    MuRILClassifier   ─┼─→ [RiskScorer.update()] ─→ RiskResult → alert
    GeminiSLM ────────┘

Design decisions:
  - RiskScorer is STATEFUL across a call. It accumulates tiers seen and
    applies temporal decay so old-call context doesn't dominate late calls.
  - Tier accumulation is a separate running set (not from the rolling
    buffer). Once Tier 1 is seen at t=10s, it stays in tiers_seen_this_call
    even when the buffer rolls forward and those utterances drop off.
  - All-3-tiers boost (+0.15): when all three tiers have been seen across
    the call lifetime, a +0.15 boost is added. This catches slow-build
    scams where tiers appear in different time windows.
  - Temporal decay: before each new fusion, the previous running score is
    multiplied by decay_factor (default 0.9). instant_escalate=True from
    heuristics bypasses decay entirely — a catastrophic phrase locks the
    score in.
  - SLM result is optional per update() cycle — MuRIL gates SLM, so most
    calls arrive here without an SLM result (SLM score defaults to 0.5 when
    absent, weighted normally — conservative).
  - Config loaded from dict (caller passes yaml config section) — no file
    I/O in this module.

Fusion formula:
    fused = 0.20 × h_score + 0.30 × c_score + 0.50 × s_score
    final = decay(prev_score) × (1 - blend) + fused × blend   [blend=1.0 for now]
    if all 3 tiers seen: final = min(1.0, final + 0.15)

Alert thresholds (from config):
    < 0.35   → SAFE
    0.35–0.65 → CAUTION
    0.65–0.85 → ALERT
    ≥ 0.85   → CRITICAL

Usage:
    scorer = RiskScorer(config)        # pass yaml config dict
    result = scorer.update(h, c, slm)  # slm may be None
    if result.should_alert:
        show_alert(result.alert_level, result.explanation)
    scorer.reset()                     # call at end of session
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from kavach.detection.heuristics  import HeuristicResult
from kavach.detection.classifier  import ClassifierResult
from kavach.detection.slm         import SLMResult

logger = logging.getLogger(__name__)

# Default weights — mirror config.yaml fusion section
_W_HEURISTIC  = 0.20
_W_CLASSIFIER = 0.30
_W_SLM        = 0.50

# Default thresholds — mirror config.yaml alerts section
_T_CAUTION  = 0.35
_T_ALERT    = 0.65
_T_CRITICAL = 0.85

# Boost when all three tiers accumulated across the call lifetime
_ALL_TIERS_BOOST = 0.15

# Default temporal decay factor
_DEFAULT_DECAY = 0.90


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskResult:
    """
    Authoritative per-cycle risk assessment from RiskScorer.update().

    Attributes:
        final_score          : 0.0–1.0, weighted fusion + decay + boost.
        alert_level          : "SAFE" | "CAUTION" | "ALERT" | "CRITICAL".
        tiers_seen_this_call : accumulated set across entire call lifetime.
        component_scores     : {"heuristic": x, "classifier": y, "slm": z}.
        should_alert         : True if alert_level != "SAFE".
        explanation          : from SLMResult.reason if available, else
                               auto-generated from tiers/heuristic matches.
    """
    final_score          : float
    alert_level          : str
    tiers_seen_this_call : List[int]
    component_scores     : Dict[str, float]
    should_alert         : bool
    explanation          : str

    def __repr__(self) -> str:
        return (
            f"RiskResult("
            f"score={self.final_score:.3f}, "
            f"level={self.alert_level!r}, "
            f"tiers={self.tiers_seen_this_call}, "
            f"alert={self.should_alert})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Scorer
# ─────────────────────────────────────────────────────────────────────────────

class RiskScorer:
    """
    Stateful weighted score fusion with tier accumulation and temporal decay.

    One instance lives for the lifetime of a call. Call reset() between calls.

    Args:
        config : dict with optional keys:
                   fusion.weight_heuristic  (default 0.20)
                   fusion.weight_classifier (default 0.30)
                   fusion.weight_slm        (default 0.50)
                   fusion.temporal_decay    (default 0.90)
                   alerts.caution           (default 0.35)
                   alerts.alert             (default 0.65)
                   alerts.critical          (default 0.85)
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        fusion = cfg.get("fusion", {})
        alerts = cfg.get("alerts", {})

        self._w_h   = float(fusion.get("weight_heuristic",  _W_HEURISTIC))
        self._w_c   = float(fusion.get("weight_classifier", _W_CLASSIFIER))
        self._w_s   = float(fusion.get("weight_slm",        _W_SLM))
        self._decay = float(fusion.get("temporal_decay",    _DEFAULT_DECAY))

        self._t_caution  = float(alerts.get("caution",  _T_CAUTION))
        self._t_alert    = float(alerts.get("alert",    _T_ALERT))
        self._t_critical = float(alerts.get("critical", _T_CRITICAL))

        # Running state (reset between calls)
        self._running_score  : float     = 0.0
        self._tiers_seen     : set[int]  = set()
        self._instant_locked : bool      = False   # True once instant_escalate fires

    # ── State management ──────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all call-lifetime state. Call at end of each call session."""
        self._running_score  = 0.0
        self._tiers_seen     = set()
        self._instant_locked = False
        logger.debug("[Scorer] State reset.")

    # ── Core update ───────────────────────────────────────────────────────────

    def update(
        self,
        heuristic  : HeuristicResult,
        classifier : ClassifierResult,
        slm        : Optional[SLMResult] = None,
    ) -> RiskResult:
        """
        Fuse the three detection signals and return the current risk level.

        Args:
            heuristic  : output of HeuristicDetector.analyze()
            classifier : output of MuRILClassifier.predict()
            slm        : output of GeminiSLM.analyze(), or None if MuRIL
                         did not escalate (SLM defaulting to 0.5 in that case).

        Returns:
            RiskResult with the updated authoritative risk score.
        """
        # ── 1. Collect scores ────────────────────────────────────────────────
        h_score = float(heuristic.heuristic_score)
        c_score = float(classifier.p_scam)
        s_score = float(slm.p_scam) if slm is not None else 0.5

        # ── 2. Tier accumulation (call-lifetime running set) ─────────────────
        self._tiers_seen.update(heuristic.tiers_detected)
        if slm is not None:
            self._tiers_seen.update(slm.tiers_detected)

        # ── 3. Temporal decay — skip if instant_escalate just fired ──────────
        if heuristic.instant_escalate:
            # Lock: instant escalate result is never diluted by decay
            self._instant_locked = True

        if self._instant_locked:
            # Once locked, keep running score as-is (don't decay it down)
            prev = self._running_score
        else:
            prev = self._running_score * self._decay

        # ── 4. Weighted fusion ───────────────────────────────────────────────
        fused = (
            self._w_h * h_score
            + self._w_c * c_score
            + self._w_s * s_score
        )

        # Blend: take the higher of (decayed prev, new fused) so score
        # never drops abruptly when a scam conversation starts mid-call
        new_score = max(prev, fused)

        # ── 5. All-3-tiers boost ─────────────────────────────────────────────
        if self._tiers_seen >= {1, 2, 3}:
            new_score = min(1.0, new_score + _ALL_TIERS_BOOST)

        new_score = min(1.0, max(0.0, new_score))
        self._running_score = new_score

        # ── 6. Alert level ───────────────────────────────────────────────────
        alert_level = self._classify(new_score)

        # ── 7. Explanation ───────────────────────────────────────────────────
        if slm is not None and slm.reason:
            explanation = slm.reason
        elif heuristic.matched_patterns:
            tier_str = f"Tier{'s' if len(heuristic.tiers_detected)>1 else ''} " \
                       f"{'+'.join(str(t) for t in heuristic.tiers_detected)} detected"
            explanation = f"{tier_str} — heuristic patterns: {', '.join(heuristic.matched_patterns[:4])}"
        else:
            explanation = "No significant signals detected."

        result = RiskResult(
            final_score          = new_score,
            alert_level          = alert_level,
            tiers_seen_this_call = sorted(self._tiers_seen),
            component_scores     = {
                "heuristic":  h_score,
                "classifier": c_score,
                "slm":        s_score,
            },
            should_alert = alert_level != "SAFE",
            explanation  = explanation,
        )
        logger.info(
            f"[Scorer] {alert_level} score={new_score:.3f} "
            f"tiers={sorted(self._tiers_seen)} "
            f"h={h_score:.2f} c={c_score:.2f} s={s_score:.2f}"
        )
        return result

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _classify(self, score: float) -> str:
        if score >= self._t_critical:
            return "CRITICAL"
        if score >= self._t_alert:
            return "ALERT"
        if score >= self._t_caution:
            return "CAUTION"
        return "SAFE"

    def __repr__(self) -> str:
        return (
            f"RiskScorer("
            f"score={self._running_score:.3f}, "
            f"tiers={sorted(self._tiers_seen)}, "
            f"locked={self._instant_locked})"
        )
