"""
kavach.detection.classifier
=============================
MuRIL binary scam classifier — fast gate before the expensive SLM.

Pipeline position:
    heuristics → [MuRILClassifier] → SLM (only if p_scam >= 0.35)

This module has two classes:

  MuRILClassifier  — inference-only, used in the live pipeline.
                     Loads model once in __init__, reuses for all calls.
                     Zero-shot until fine-tuned; plumbing works immediately.
                     Graceful fallback (p_scam=0.5) if model fails to load.

  MuRILTrainer     — fine-tuning scaffold, run offline once dataset is ready.
                     Reads a CSV (text, label columns), performs train/test
                     split, fine-tunes the classification head, saves weights.
                     If no conversation_id column exists, falls back to
                     row-level split with a warning (BTP-I dataset compat).

Input:
    buf.as_classifier_input() — full conversation string, both speakers,
    CALLER/USER labels included.  e.g.:
        "CALLER: i am from sbi USER: yes who is this CALLER: share otp"

Output:
    ClassifierResult with p_scam, is_suspicious, escalate_to_slm, timing.

Threshold defaults:
    safe_threshold      = 0.35   (below → skip SLM)
    suspicious_threshold = 0.65  (above → strong signal for SLM)

Design decisions:
  - Model loaded once in __init__ via _load_model(). Thread-safe read (no
    writes after init) so it can be shared across the pipeline safely.
  - zero-shot accuracy is intentionally low — the classification head starts
    with random weights. Fine-tune with MuRILTrainer to make this meaningful.
  - Conversation-level split in trainer is critical: row-level splitting of
    turn-segmented conversations leaks context across train/test boundary.
    (This was a confirmed accuracy inflation bug in BTP-I.)
  - torch.no_grad() used in predict() — never accumulate gradients at runtime.
  - sentencepiece required by MuRIL tokenizer — must be installed.

Usage (inference):
    from kavach.detection.classifier import MuRILClassifier
    clf = MuRILClassifier()
    result = clf.predict(buf.as_classifier_input())
    if result.escalate_to_slm:
        slm_result = slm.analyze(buf.as_slm_context())

Usage (fine-tuning, offline):
    from kavach.detection.classifier import MuRILTrainer
    trainer = MuRILTrainer(output_dir="models/muril_finetuned")
    trainer.train("data/final_scam_dataset.csv", epochs=3)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "google/muril-base-cased"
_MAX_LENGTH = 256
_DEFAULT_THRESHOLD = 0.35


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClassifierResult:
    """
    Output of MuRILClassifier.predict().

    Attributes:
        p_scam          : probability of scam (0.0–1.0).
        is_suspicious   : True if p_scam >= threshold.
        escalate_to_slm : True if p_scam >= threshold (same gate for now;
                          future: could use a separate higher threshold).
        inference_ms    : wall-clock time for this inference call (ms).
    """
    p_scam          : float
    is_suspicious   : bool
    escalate_to_slm : bool
    inference_ms    : float

    def __repr__(self) -> str:
        return (
            f"ClassifierResult("
            f"p_scam={self.p_scam:.3f}, "
            f"suspicious={self.is_suspicious}, "
            f"escalate={self.escalate_to_slm}, "
            f"ms={self.inference_ms:.1f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────

class MuRILClassifier:
    """
    MuRIL-based binary scam classifier.

    Loads google/muril-base-cased (or a fine-tuned local checkpoint) once
    at init. All subsequent predict() calls reuse the loaded model.

    Until fine-tuned, the classification head has random weights — p_scam
    will be near 0.5 for all inputs. That is expected and correct. The
    plumbing (tokenisation → forward pass → softmax → threshold gate) works
    end-to-end immediately.

    Args:
        model_path  : HuggingFace model ID or path to local checkpoint.
                      Default: 'google/muril-base-cased'.
        threshold   : p_scam cutoff for is_suspicious / escalate_to_slm.
                      Default: 0.35 (from config classifier.safe_threshold).
        device      : 'cpu' or 'cuda'. Default: 'cpu'.
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL,
        threshold: float = _DEFAULT_THRESHOLD,
        device: str = "cpu",
    ) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self.device = device
        self._model = None
        self._tokenizer = None
        self._fallback_mode = False

        self._load_model()

    def _load_model(self) -> None:
        """
        Load tokenizer and model. Sets _fallback_mode=True on any failure
        so predict() can continue without crashing the pipeline.
        """
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
            import torch

            logger.info(f"[Classifier] Loading MuRIL from '{self.model_path}'...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=2,
                ignore_mismatched_sizes=True,   # head size mismatch is expected
            )
            self._model.to(self.device)
            self._model.eval()
            logger.info("[Classifier] MuRIL loaded successfully.")

        except Exception as e:
            logger.warning(
                f"[Classifier] Failed to load MuRIL model: {e}\n"
                "Falling back to p_scam=0.5 — SLM will decide on all inputs."
            )
            self._fallback_mode = True

    def predict(self, text: str) -> ClassifierResult:
        """
        Run the classifier on a conversation string.

        Args:
            text : output of buf.as_classifier_input() — full conversation
                   with CALLER/USER labels, already lowercased.

        Returns:
            ClassifierResult. If model failed to load, returns p_scam=0.5
            (conservative: pass everything to SLM rather than silently drop).
        """
        t0 = time.perf_counter()

        if self._fallback_mode or self._model is None or self._tokenizer is None:
            logger.debug("[Classifier] Fallback mode — returning p_scam=0.5.")
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return ClassifierResult(
                p_scam=0.5,
                is_suspicious=True,          # err on the side of caution
                escalate_to_slm=True,
                inference_ms=elapsed_ms,
            )

        try:
            import torch

            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=_MAX_LENGTH,
                padding=True,
            )
            # Move tensors to same device as model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            probs = torch.softmax(outputs.logits, dim=-1)
            p_scam = float(probs[0][1].item())

        except Exception as e:
            logger.warning(f"[Classifier] Inference failed: {e} — returning p_scam=0.5.")
            p_scam = 0.5

        elapsed_ms = (time.perf_counter() - t0) * 1000
        is_suspicious = p_scam >= self.threshold

        return ClassifierResult(
            p_scam=p_scam,
            is_suspicious=is_suspicious,
            escalate_to_slm=is_suspicious,
            inference_ms=elapsed_ms,
        )

    def __repr__(self) -> str:
        return (
            f"MuRILClassifier("
            f"model='{self.model_path}', "
            f"threshold={self.threshold}, "
            f"device='{self.device}', "
            f"fallback={self._fallback_mode})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Trainer (offline fine-tuning scaffold)
# ─────────────────────────────────────────────────────────────────────────────

class MuRILTrainer:
    """
    Fine-tuning scaffold for MuRILClassifier.

    NOT used in the live pipeline. Run this offline once the scam dataset
    is ready. Saves the fine-tuned model to output_dir, which is then
    passed as model_path to MuRILClassifier.

    Train/test split strategy:
      - If CSV has a 'conversation_id' column: split on unique conversation
        IDs (group-aware split) to prevent context leakage across turns.
      - If not: fall back to row-level split with a warning.
        Row-level split inflates accuracy when conversations are multi-turn
        (confirmed bug in BTP-I dataset). Add conversation_id when rebuilding.

    Args:
        base_model  : HuggingFace model ID for the base MuRIL weights.
        output_dir  : directory to save fine-tuned weights and tokenizer.

    Usage:
        trainer = MuRILTrainer(output_dir="models/muril_finetuned")
        trainer.train("data/final_scam_dataset.csv", epochs=3, batch_size=8)
    """

    def __init__(
        self,
        base_model: str = _DEFAULT_MODEL,
        output_dir: str = "models/muril_finetuned",
    ) -> None:
        self.base_model = base_model
        self.output_dir = output_dir

    def train(
        self,
        csv_path: str,
        test_size: float = 0.20,
        epochs: int = 3,
        batch_size: int = 8,
    ) -> None:
        """
        Fine-tune the MuRIL classification head on a labelled CSV.

        Args:
            csv_path   : path to CSV with 'text' and 'label' columns.
                         Optional 'conversation_id' for group-aware split.
            test_size  : fraction of data to hold out for evaluation.
            epochs     : number of training epochs.
            batch_size : per-device batch size.
        """
        import os
        import numpy as np
        import pandas as pd
        from sklearn.metrics import classification_report
        from sklearn.model_selection import GroupShuffleSplit, train_test_split
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
        import torch
        from torch.utils.data import Dataset

        # ── Load data ────────────────────────────────────────────────────────
        logger.info(f"[Trainer] Loading dataset from '{csv_path}'...")
        df = pd.read_csv(csv_path)

        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError(
                f"CSV must have 'text' and 'label' columns. "
                f"Found: {list(df.columns)}"
            )

        # ── Train / test split ───────────────────────────────────────────────
        if "conversation_id" in df.columns:
            logger.info("[Trainer] Splitting on conversation_id (group-aware).")
            groups = df["conversation_id"].values
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
            train_idx, test_idx = next(splitter.split(df, groups=groups))
            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df  = df.iloc[test_idx].reset_index(drop=True)
        else:
            logger.warning(
                "[Trainer] No 'conversation_id' column found — falling back to "
                "row-level split. This may inflate accuracy on multi-turn datasets. "
                "Add conversation_id when rebuilding the dataset."
            )
            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=42, stratify=df["label"]
            )
            train_df = train_df.reset_index(drop=True)
            test_df  = test_df.reset_index(drop=True)

        logger.info(
            f"[Trainer] Train: {len(train_df)} rows, Test: {len(test_df)} rows"
        )

        # ── Load tokenizer + model ───────────────────────────────────────────
        logger.info(f"[Trainer] Loading base model '{self.base_model}'...")
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )

        # ── Tokenize ─────────────────────────────────────────────────────────
        class _ScamDataset(Dataset):
            def __init__(self, texts, labels, tok):
                encodings = tok(
                    list(texts),
                    truncation=True,
                    max_length=_MAX_LENGTH,
                    padding="max_length",
                )
                self.input_ids      = encodings["input_ids"]
                self.attention_mask = encodings["attention_mask"]
                self.labels         = list(labels)

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                return {
                    "input_ids":      torch.tensor(self.input_ids[idx],      dtype=torch.long),
                    "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
                    "labels":         torch.tensor(self.labels[idx],         dtype=torch.long),
                }

        train_dataset = _ScamDataset(train_df["text"], train_df["label"], tokenizer)
        eval_dataset  = _ScamDataset(test_df["text"],  test_df["label"],  tokenizer)

        # ── Training arguments ───────────────────────────────────────────────
        args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            logging_steps=50,
            seed=42,
            no_cuda=True,           # CPU training for now; remove for GPU
        )

        # ── Metrics ──────────────────────────────────────────────────────────
        def _compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            report = classification_report(
                labels, preds,
                target_names=["legit", "scam"],
                output_dict=True,
                zero_division=0,
            )
            return {
                "precision_scam": report["scam"]["precision"],
                "recall_scam":    report["scam"]["recall"],
                "f1_scam":        report["scam"]["f1-score"],
                "f1_macro":       report["macro avg"]["f1-score"],
            }

        # ── Train ────────────────────────────────────────────────────────────
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=_compute_metrics,
        )

        logger.info(f"[Trainer] Starting fine-tuning ({epochs} epochs)...")
        trainer.train()

        # ── Evaluation report ────────────────────────────────────────────────
        logger.info("[Trainer] Running final evaluation...")
        preds_output = trainer.predict(eval_dataset)
        preds = np.argmax(preds_output.predictions, axis=-1)
        true  = test_df["label"].values

        print("\n" + "="*55)
        print("  MuRIL Fine-tuning — Evaluation Results")
        print("="*55)
        print(classification_report(
            true, preds,
            target_names=["legit", "scam"],
            zero_division=0,
        ))
        print("="*55 + "\n")

        # ── Save ─────────────────────────────────────────────────────────────
        os.makedirs(self.output_dir, exist_ok=True)
        trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        logger.info(f"[Trainer] Model saved to '{self.output_dir}'.")
