"""
tests/test_terminal_ui.py
==========================
Unit tests for kavach.ui.terminal_ui — logic methods only.
No visual/Live output tested; all assertions are on return values.

Coverage:
  1.  _make_risk_bar(0.0)  → all empty blocks
  2.  _make_risk_bar(1.0)  → all filled blocks
  3.  _make_risk_bar(0.5)  → exactly half filled
  4.  _make_risk_bar clamps values outside [0, 1]
  5.  _make_risk_bar total length always equals BAR_WIDTH
  6.  _alert_color("SAFE")     → "green"
  7.  _alert_color("CAUTION")  → "yellow"
  8.  _alert_color("ALERT")    → "bold red"
  9.  _alert_color("CRITICAL") → "bold red"
  10. _format_utterance truncates at 60 chars
  11. _format_utterance CALLER gets yellow style
  12. _format_utterance USER gets blue style
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kavach.ui.terminal_ui import (
    _make_risk_bar,
    _alert_color,
    _format_utterance,
    _BAR_WIDTH,
)
from kavach.audio.buffer import Speaker


# ── 1–5: _make_risk_bar ───────────────────────────────────────────────────────

def test_risk_bar_zero_all_empty():
    bar = _make_risk_bar(0.0)
    assert bar == "░" * _BAR_WIDTH
    assert "█" not in bar


def test_risk_bar_one_all_filled():
    bar = _make_risk_bar(1.0)
    assert bar == "█" * _BAR_WIDTH
    assert "░" not in bar


def test_risk_bar_half():
    bar = _make_risk_bar(0.5)
    half = _BAR_WIDTH // 2
    assert bar == "█" * half + "░" * half


def test_risk_bar_clamps_below_zero():
    bar = _make_risk_bar(-5.0)
    assert bar == "░" * _BAR_WIDTH


def test_risk_bar_clamps_above_one():
    bar = _make_risk_bar(99.0)
    assert bar == "█" * _BAR_WIDTH


def test_risk_bar_total_length_always_bar_width():
    for score in (0.0, 0.1, 0.33, 0.5, 0.73, 0.99, 1.0):
        bar = _make_risk_bar(score)
        assert len(bar) == _BAR_WIDTH, f"len={len(bar)} for score={score}"


# ── 6–9: _alert_color ─────────────────────────────────────────────────────────

def test_alert_color_safe():
    assert _alert_color("SAFE") == "green"


def test_alert_color_caution():
    assert _alert_color("CAUTION") == "yellow"


def test_alert_color_alert():
    assert _alert_color("ALERT") == "bold red"


def test_alert_color_critical():
    assert _alert_color("CRITICAL") == "bold red"


# ── 10–12: _format_utterance ──────────────────────────────────────────────────

def test_format_utterance_truncates_at_60_chars():
    long_text = "a" * 80
    line = _format_utterance(Speaker.CALLER, long_text, 0.0)
    plain = line.plain
    # The displayed text portion should be truncated — "…" at the end
    assert "…" in plain
    # Total visible text chars after the label prefix should be ≤ 61 (60 + ellipsis)
    # Just confirm the full 80-char string is NOT present
    assert "a" * 80 not in plain


def test_format_utterance_short_text_not_truncated():
    text = "short text"
    line = _format_utterance(Speaker.CALLER, text, 5.0)
    assert text in line.plain
    assert "…" not in line.plain


def test_format_utterance_caller_is_yellow():
    line = _format_utterance(Speaker.CALLER, "hello", 0.0)
    # Inspect the spans — at least one span should have yellow style
    styles = [str(span.style) for span in line._spans]
    assert any("yellow" in s for s in styles)


def test_format_utterance_user_is_blue():
    line = _format_utterance(Speaker.USER, "hello", 0.0)
    styles = [str(span.style) for span in line._spans]
    assert any("blue" in s for s in styles)
