"""
tests/test_classifier.py
=========================
Unit tests for kavach.detection.classifier.

All HuggingFace model/tokenizer calls are mocked — no network, no downloads.
Tests cover:
  - predict() returns correct types
  - threshold logic (0.34 → not suspicious, 0.36 → suspicious)
  - escalate_to_slm matches is_suspicious
  - inference_ms is positive
  - graceful fallback when model unavailable
  - fallback returns p_scam=0.5 and escalates conservatively
  - custom threshold respected
  - empty string input doesn't crash
  - MuRILTrainer.train() runs with mocked transformers (no download)
  - Trainer falls back to row-level split when no conversation_id
  - Trainer uses group split when conversation_id present
  - ClassifierResult __repr__ works
  - predict() on scam-like text (mocked to return high p_scam)
  - predict() on safe text (mocked to return low p_scam)
"""

import sys
import os
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kavach.detection.classifier import ClassifierResult, MuRILClassifier


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — build a mock that makes MuRILClassifier load without HuggingFace
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_outputs(p_scam: float = 0.7):
    """Return a mock model output whose logits give the desired p_scam."""
    import torch
    # inverse softmax: if we want p_scam=p, logits[1]-logits[0] = log(p/(1-p))
    # simplest: just use [0, 1] logits scaled so softmax ≈ p_scam
    import math
    log_odds = math.log(p_scam / (1 - p_scam))
    logits = torch.tensor([[0.0, log_odds]])
    mock_output = MagicMock()
    mock_output.logits = logits
    return mock_output


def _mock_tokenizer_call():
    """Return a mock tokenizer callable that produces valid tensor-like output."""
    import torch
    mock_tok = MagicMock()
    mock_tok.return_value = {
        "input_ids":      torch.zeros(1, 10, dtype=torch.long),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
    }
    return mock_tok


def _make_classifier(p_scam: float = 0.7, threshold: float = 0.35) -> MuRILClassifier:
    """
    Build a MuRILClassifier with fully mocked tokenizer and model.
    Model always returns the given p_scam value.
    """
    mock_tok = _mock_tokenizer_call()
    mock_model = MagicMock()
    mock_model.return_value = _make_mock_outputs(p_scam)
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok), \
         patch("transformers.AutoModelForSequenceClassification.from_pretrained",
               return_value=mock_model):
        clf = MuRILClassifier(threshold=threshold)

    return clf


# ─────────────────────────────────────────────────────────────────────────────
# 1. predict() returns ClassifierResult with correct types
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_returns_classifier_result():
    clf = _make_classifier(p_scam=0.7)
    result = clf.predict("CALLER: share your otp USER: ok")
    assert isinstance(result, ClassifierResult)
    assert isinstance(result.p_scam, float)
    assert isinstance(result.is_suspicious, bool)
    assert isinstance(result.escalate_to_slm, bool)
    assert isinstance(result.inference_ms, float)


# ─────────────────────────────────────────────────────────────────────────────
# 2. p_scam is within [0, 1]
# ─────────────────────────────────────────────────────────────────────────────

def test_p_scam_in_range():
    clf = _make_classifier(p_scam=0.7)
    result = clf.predict("any text")
    assert 0.0 <= result.p_scam <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Threshold: p_scam=0.34 → not suspicious
# ─────────────────────────────────────────────────────────────────────────────

def test_threshold_below_not_suspicious():
    clf = _make_classifier(p_scam=0.34, threshold=0.35)
    result = clf.predict("hello this is a courtesy call")
    assert result.is_suspicious is False
    assert result.escalate_to_slm is False


# ─────────────────────────────────────────────────────────────────────────────
# 4. Threshold: p_scam=0.36 → suspicious
# ─────────────────────────────────────────────────────────────────────────────

def test_threshold_above_suspicious():
    clf = _make_classifier(p_scam=0.36, threshold=0.35)
    result = clf.predict("calling from sbi share otp")
    assert result.is_suspicious is True
    assert result.escalate_to_slm is True


# ─────────────────────────────────────────────────────────────────────────────
# 5. escalate_to_slm matches is_suspicious
# ─────────────────────────────────────────────────────────────────────────────

def test_escalate_matches_is_suspicious_high():
    clf = _make_classifier(p_scam=0.8)
    result = clf.predict("arrest warrant issued against your name")
    assert result.escalate_to_slm == result.is_suspicious


def test_escalate_matches_is_suspicious_low():
    clf = _make_classifier(p_scam=0.2)
    result = clf.predict("your statement is ready visit the branch")
    assert result.escalate_to_slm == result.is_suspicious


# ─────────────────────────────────────────────────────────────────────────────
# 6. inference_ms is positive
# ─────────────────────────────────────────────────────────────────────────────

def test_inference_ms_positive():
    clf = _make_classifier(p_scam=0.6)
    result = clf.predict("some text")
    assert result.inference_ms > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. Graceful fallback when model fails to load
# ─────────────────────────────────────────────────────────────────────────────

def test_graceful_fallback_on_load_failure():
    with patch(
        "transformers.AutoTokenizer.from_pretrained",
        side_effect=OSError("network unavailable"),
    ):
        clf = MuRILClassifier(model_path="google/muril-base-cased")

    assert clf._fallback_mode is True
    result = clf.predict("any text")
    assert result.p_scam == pytest.approx(0.5)
    assert result.is_suspicious is True   # conservative: escalate on fallback
    assert result.escalate_to_slm is True


# ─────────────────────────────────────────────────────────────────────────────
# 8. Fallback p_scam=0.5 regardless of input
# ─────────────────────────────────────────────────────────────────────────────

def test_fallback_always_0_5():
    with patch(
        "transformers.AutoTokenizer.from_pretrained",
        side_effect=RuntimeError("model not found"),
    ):
        clf = MuRILClassifier()

    for text in ["", "hello", "otp share karo arrest warrant"]:
        result = clf.predict(text)
        assert result.p_scam == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Custom threshold respected
# ─────────────────────────────────────────────────────────────────────────────

def test_custom_threshold_065():
    clf = _make_classifier(p_scam=0.60, threshold=0.65)
    result = clf.predict("some scam-like text")
    # p_scam=0.60 < 0.65 threshold → not suspicious
    assert result.is_suspicious is False


def test_custom_threshold_020():
    clf = _make_classifier(p_scam=0.25, threshold=0.20)
    result = clf.predict("some text")
    # p_scam=0.25 >= 0.20 threshold → suspicious
    assert result.is_suspicious is True


# ─────────────────────────────────────────────────────────────────────────────
# 10. Empty string doesn't crash
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_string_no_crash():
    clf = _make_classifier(p_scam=0.5)
    result = clf.predict("")
    assert isinstance(result, ClassifierResult)


# ─────────────────────────────────────────────────────────────────────────────
# 11. ClassifierResult __repr__ works
# ─────────────────────────────────────────────────────────────────────────────

def test_classifier_result_repr():
    r = ClassifierResult(p_scam=0.75, is_suspicious=True,
                         escalate_to_slm=True, inference_ms=42.3)
    s = repr(r)
    assert "0.750" in s
    assert "True" in s


# ─────────────────────────────────────────────────────────────────────────────
# 12. MuRILTrainer.train() runs with mocked Trainer — row-level fallback
# ─────────────────────────────────────────────────────────────────────────────

def test_trainer_row_level_fallback(tmp_path, caplog):
    """
    Trainer should fall back to row-level split with a warning when CSV
    has no conversation_id column, and complete without error.
    """
    import logging
    import numpy as np
    import pandas as pd
    # Force actual import of the submodule so patch.object works (bypasses
    # transformers 5.x lazy __getattr__ loading).
    import transformers.trainer as _trainer_mod
    import transformers.training_args as _args_mod
    from kavach.detection.classifier import MuRILTrainer

    # Minimal CSV: no conversation_id
    csv_path = tmp_path / "test.csv"
    pd.DataFrame({
        "text":  ["hello " * 10] * 40,
        "label": ([0] * 20) + ([1] * 20),
    }).to_csv(csv_path, index=False)

    mock_tok = MagicMock()
    mock_tok.return_value = {
        "input_ids":      [[0] * 256] * 40,
        "attention_mask": [[1] * 256] * 40,
    }

    mock_trainer_instance = MagicMock()
    mock_trainer_instance.train.return_value = None
    mock_trainer_instance.predict.return_value = MagicMock(
        predictions=np.array([[0.8, 0.2]] * 8)   # 20% of 40
    )

    with patch("transformers.AutoTokenizer.from_pretrained",      return_value=mock_tok), \
         patch("transformers.AutoModelForSequenceClassification.from_pretrained",
               return_value=MagicMock()), \
         patch.object(_trainer_mod, "Trainer", return_value=mock_trainer_instance), \
         patch.object(_args_mod,   "TrainingArguments", return_value=MagicMock()), \
         patch("os.makedirs"), \
         patch.object(mock_tok, "save_pretrained"), \
         caplog.at_level(logging.WARNING, logger="kavach.detection.classifier"):

        trainer_obj = MuRILTrainer(output_dir=str(tmp_path / "model"))
        trainer_obj.train(str(csv_path), epochs=1, batch_size=8)

    mock_trainer_instance.train.assert_called_once()
    assert "row-level split" in caplog.text


# ─────────────────────────────────────────────────────────────────────────────
# 13. MuRILTrainer uses group split when conversation_id present
# ─────────────────────────────────────────────────────────────────────────────

def test_trainer_group_split_used(tmp_path, caplog):
    """
    When conversation_id is present, trainer should NOT emit the row-level
    fallback warning.
    """
    import logging
    import numpy as np
    import pandas as pd
    import transformers.trainer as _trainer_mod
    import transformers.training_args as _args_mod
    from kavach.detection.classifier import MuRILTrainer

    csv_path = tmp_path / "test_group.csv"
    pd.DataFrame({
        "text":            ["hello " * 10] * 40,
        "label":           ([0] * 20) + ([1] * 20),
        "conversation_id": list(range(40)),   # unique id per row → group split
    }).to_csv(csv_path, index=False)

    mock_trainer_instance = MagicMock()
    mock_trainer_instance.train.return_value = None
    mock_trainer_instance.predict.return_value = MagicMock(
        predictions=np.array([[0.8, 0.2]] * 8)
    )

    mock_tok = MagicMock()
    mock_tok.return_value = {
        "input_ids":      [[0] * 256] * 40,
        "attention_mask": [[1] * 256] * 40,
    }

    with patch("transformers.AutoTokenizer.from_pretrained",      return_value=mock_tok), \
         patch("transformers.AutoModelForSequenceClassification.from_pretrained",
               return_value=MagicMock()), \
         patch.object(_trainer_mod, "Trainer", return_value=mock_trainer_instance), \
         patch.object(_args_mod,   "TrainingArguments", return_value=MagicMock()), \
         patch("os.makedirs"), \
         patch.object(mock_tok, "save_pretrained"), \
         caplog.at_level(logging.WARNING, logger="kavach.detection.classifier"):

        trainer = MuRILTrainer(output_dir=str(tmp_path / "model"))
        trainer.train(str(csv_path), epochs=1, batch_size=8)

    # The row-level fallback warning should NOT appear
    assert "row-level split" not in caplog.text
