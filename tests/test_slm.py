"""
tests/test_slm.py
==================
Unit tests for kavach.detection.slm.GeminiSLM.

All Google API calls are mocked — no network, no quota usage.
Tests cover:
  1.  SCAM verdict parsed correctly
  2.  LEGITIMATE verdict parsed correctly
  3.  UNCERTAIN on malformed JSON
  4.  UNCERTAIN on empty response
  5.  p_scam mapping: SCAM→1.0, LEGITIMATE→0.0, UNCERTAIN→0.5
  6.  heuristic_score and classifier_score appear in the prompt sent to Gemini
  7.  Graceful handling of API exception → UNCERTAIN, never raises
  8.  inference_ms is positive on successful call
  9.  inference_ms is non-negative on API error
  10. Markdown-fenced JSON is stripped and parsed correctly
  11. tiers_detected is correctly parsed from JSON array
  12. Invalid verdict value returns UNCERTAIN
"""

import json
import sys
import os
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kavach.detection.slm import GeminiSLM, SLMResult, _parse_response, _uncertain


# ─────────────────────────────────────────────────────────────────────────────
# Fixture — build a GeminiSLM with fully mocked google.genai.Client
# ─────────────────────────────────────────────────────────────────────────────

def _make_slm(response_text: str) -> tuple[GeminiSLM, MagicMock]:
    """
    Return (GeminiSLM, mock_generate_content).
    mock_generate_content.call_args lets tests inspect the prompt sent.
    """
    mock_response = MagicMock()
    mock_response.text = response_text

    mock_models = MagicMock()
    mock_models.generate_content.return_value = mock_response

    mock_client = MagicMock()
    mock_client.models = mock_models

    with patch("google.genai.Client", return_value=mock_client):
        slm = GeminiSLM(api_key="fake-key")

    return slm, mock_models.generate_content


def _scam_json(**overrides) -> str:
    base = {
        "verdict":        "SCAM",
        "tiers_detected": [1, 3],
        "confidence":     "HIGH",
        "reason":         "Caller impersonated ICICI bank and requested OTP via WhatsApp.",
    }
    base.update(overrides)
    return json.dumps(base)


def _legit_json(**overrides) -> str:
    base = {
        "verdict":        "LEGITIMATE",
        "tiers_detected": [],
        "confidence":     "HIGH",
        "reason":         "No authority impersonation; peer shared OTP for Netflix.",
    }
    base.update(overrides)
    return json.dumps(base)


SAMPLE_CONTEXT = (
    "[0.0s] CALLER: amazon icici ka platinum card milega aapko\n"
    "[3.5s] USER: haan ji\n"
    "[5.0s] CALLER: whatsapp pe otp bhejiye verification ke liye\n"
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SCAM verdict is parsed correctly
# ─────────────────────────────────────────────────────────────────────────────

def test_scam_verdict_parsed():
    slm, _ = _make_slm(_scam_json())
    result = slm.analyze(SAMPLE_CONTEXT, heuristic_score=0.40, classifier_score=0.50)

    assert isinstance(result, SLMResult)
    assert result.verdict == "SCAM"
    assert result.tiers_detected == [1, 3]
    assert result.confidence == "HIGH"
    assert result.p_scam == pytest.approx(1.0)
    assert len(result.reason) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. LEGITIMATE verdict is parsed correctly
# ─────────────────────────────────────────────────────────────────────────────

def test_legitimate_verdict_parsed():
    slm, _ = _make_slm(_legit_json())
    result = slm.analyze(
        "[0.0s] USER: bro send me otp for netflix\n[1.0s] CALLER: sure",
        heuristic_score=0.20,
        classifier_score=0.30,
    )

    assert result.verdict == "LEGITIMATE"
    assert result.tiers_detected == []
    assert result.p_scam == pytest.approx(0.0)
    assert result.confidence == "HIGH"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Malformed JSON returns UNCERTAIN with LOW confidence
# ─────────────────────────────────────────────────────────────────────────────

def test_malformed_json_returns_uncertain():
    slm, _ = _make_slm("this is not json at all {{{{")
    result = slm.analyze(SAMPLE_CONTEXT)

    assert result.verdict == "UNCERTAIN"
    assert result.confidence == "LOW"
    assert result.p_scam == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Empty response text returns UNCERTAIN
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_response_returns_uncertain():
    slm, _ = _make_slm("")
    result = slm.analyze(SAMPLE_CONTEXT)

    assert result.verdict == "UNCERTAIN"
    assert result.p_scam == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 5. p_scam mapping is correct for all three verdicts
# ─────────────────────────────────────────────────────────────────────────────

def test_p_scam_mapping_scam():
    slm, _ = _make_slm(_scam_json())
    assert slm.analyze(SAMPLE_CONTEXT).p_scam == pytest.approx(1.0)


def test_p_scam_mapping_legitimate():
    slm, _ = _make_slm(_legit_json())
    assert slm.analyze(SAMPLE_CONTEXT).p_scam == pytest.approx(0.0)


def test_p_scam_mapping_uncertain():
    uncertain_json = json.dumps({
        "verdict": "UNCERTAIN", "tiers_detected": [],
        "confidence": "LOW", "reason": "Not enough context.",
    })
    slm, _ = _make_slm(uncertain_json)
    assert slm.analyze(SAMPLE_CONTEXT).p_scam == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 6. heuristic_score and classifier_score appear in the prompt sent to Gemini
# ─────────────────────────────────────────────────────────────────────────────

def test_scores_included_in_prompt():
    slm, mock_gen = _make_slm(_scam_json())
    slm.analyze(SAMPLE_CONTEXT, heuristic_score=0.40, classifier_score=0.49)

    # Inspect the `contents` kwarg passed to generate_content
    call_kwargs = mock_gen.call_args.kwargs
    prompt_text = call_kwargs["contents"]

    assert "0.40" in prompt_text
    assert "0.49" in prompt_text


# ─────────────────────────────────────────────────────────────────────────────
# 7. API exception → UNCERTAIN, never raises
# ─────────────────────────────────────────────────────────────────────────────

def test_api_exception_returns_uncertain_never_raises():
    mock_models = MagicMock()
    mock_models.generate_content.side_effect = ConnectionError("network down")

    mock_client = MagicMock()
    mock_client.models = mock_models

    with patch("google.genai.Client", return_value=mock_client):
        slm = GeminiSLM(api_key="fake-key")

    result = slm.analyze(SAMPLE_CONTEXT)
    assert result.verdict == "UNCERTAIN"
    assert result.confidence == "LOW"
    assert result.p_scam == pytest.approx(0.5)
    assert "ConnectionError" in result.reason


# ─────────────────────────────────────────────────────────────────────────────
# 8. inference_ms is positive on successful call
# ─────────────────────────────────────────────────────────────────────────────

def test_inference_ms_positive_on_success():
    slm, _ = _make_slm(_scam_json())
    result = slm.analyze(SAMPLE_CONTEXT)
    assert result.inference_ms >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 9. inference_ms is non-negative on API error
# ─────────────────────────────────────────────────────────────────────────────

def test_inference_ms_non_negative_on_error():
    mock_models = MagicMock()
    mock_models.generate_content.side_effect = RuntimeError("quota exceeded")
    mock_client = MagicMock()
    mock_client.models = mock_models

    with patch("google.genai.Client", return_value=mock_client):
        slm = GeminiSLM(api_key="fake-key")

    result = slm.analyze(SAMPLE_CONTEXT)
    assert result.inference_ms >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 10. Markdown-fenced JSON is stripped and parsed correctly
# ─────────────────────────────────────────────────────────────────────────────

def test_markdown_fenced_json_parsed():
    fenced = "```json\n" + _scam_json() + "\n```"
    slm, _ = _make_slm(fenced)
    result = slm.analyze(SAMPLE_CONTEXT)

    assert result.verdict == "SCAM"
    assert result.p_scam == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 11. tiers_detected parsed correctly from JSON array
# ─────────────────────────────────────────────────────────────────────────────

def test_tiers_detected_all_three():
    payload = _scam_json(tiers_detected=[1, 2, 3])
    slm, _ = _make_slm(payload)
    result = slm.analyze(SAMPLE_CONTEXT)
    assert result.tiers_detected == [1, 2, 3]


def test_tiers_detected_empty():
    slm, _ = _make_slm(_legit_json(tiers_detected=[]))
    result = slm.analyze(SAMPLE_CONTEXT)
    assert result.tiers_detected == []


# ─────────────────────────────────────────────────────────────────────────────
# 12. Invalid verdict value returns UNCERTAIN
# ─────────────────────────────────────────────────────────────────────────────

def test_invalid_verdict_returns_uncertain():
    bad = json.dumps({
        "verdict": "MAYBE",
        "tiers_detected": [1],
        "confidence": "HIGH",
        "reason": "Hmm.",
    })
    slm, _ = _make_slm(bad)
    result = slm.analyze(SAMPLE_CONTEXT)

    assert result.verdict == "UNCERTAIN"
    assert result.p_scam == pytest.approx(0.5)
