"""
kavach.detection.slm
======================
SLM reasoning layer — the final, authoritative scam verdict.

Two backends are available:
  - GeminiSLM    : Gemini Flash API (demo/eval — audio-free, but cloud)
  - OllamaLlamaSLM : Llama 3.2 3B via local Ollama server (fully on-device)

Pipeline position:
    heuristics → MuRILClassifier (gate) → [GeminiSLM | OllamaLlamaSLM] → risk_scorer

This is the ONLY layer that understands intent. It solves the false-positive
problem that pure heuristics and classifiers cannot:

  "bro share OTP for Netflix"   → Tier 3 present, but NO Tier 1 → LEGITIMATE
  "RBI officer, share OTP now"  → Tier 1 + Tier 3 in CALLER speech → SCAM

The Manipulation Funnel system prompt is embedded here as a module-level
constant shared by both classes. It cannot be overridden at runtime.

Design decisions:
  - temperature=0.1 for highly consistent structured JSON output.
  - heuristic_score and classifier_score are passed as "prior signals" in
    the user prompt — the model uses them as soft hints, not hard rules.
  - JSON is parsed strictly. Any malformed response returns UNCERTAIN/LOW
    rather than crashing the pipeline.
  - API/connection errors return UNCERTAIN so the pipeline degrades
    gracefully — it never halts a live call.
  - GeminiSLM uses google-genai (new SDK), same as GeminiASR.
  - OllamaLlamaSLM uses the ollama Python client; requires `ollama serve`.

Usage:
    from kavach.detection.slm import GeminiSLM, OllamaLlamaSLM, check_ollama_running

    # Gemini (cloud)
    slm = GeminiSLM(api_key="AIza...")

    # Ollama (on-device)
    if check_ollama_running():
        slm = OllamaLlamaSLM(model="llama3.2:3b")

    result = slm.analyze(
        slm_context=buf.as_slm_context(),
        heuristic_score=0.40,
        classifier_score=0.50,
    )
    print(result.verdict, result.confidence, result.reason)
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Manipulation Funnel system prompt — embedded verbatim, never changed at runtime
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a real-time banking scam call detector for India.
Analyze the phone conversation below and determine if it is a scam.

A call is a SCAM only if the CALLER (not the user) shows evidence of
the Manipulation Funnel pattern:

TIER 1 — Authority/Impersonation: Caller claims to be from a bank
(SBI, HDFC, ICICI, Axis, RBI), police, CBI, income tax, courier
(FedEx, Amazon, Flipkart), or any government body WITHOUT being asked.

TIER 2 — Urgency/Threat: Caller creates time pressure or threatens
consequences (arrest, account freeze, legal action, disconnection).

TIER 3 — Credential Extraction: Caller requests OTP, PIN, CVV,
full account number, Aadhaar, password, or asks victim to transfer
money or share details via WhatsApp.

VERDICT RULES:
- SCAM: Tier 1 + Tier 3 both present in CALLER speech
- SCAM: Any instant-escalate phrase present (arrest warrant, drug case,
  transfer to safe account)
- LEGITIMATE: Tier 3 present but NO Tier 1 (peer OTP sharing, own bank
  app, purchase verification — user-initiated context)
- LEGITIMATE: Bank calls that only confirm last 4 digits, schedule
  appointments, or send notifications — no credential extraction
- UNCERTAIN: Insufficient context to decide

Respond ONLY with valid JSON. No markdown, no explanation outside JSON.
Return ONLY this exact JSON structure, no other keys:
{"verdict": "SCAM", "tiers_detected": [1, 3], "confidence": "HIGH", "reason": "one sentence explanation"}\
"""

_RESPONSE_SCHEMA = """\
{
  "verdict": "SCAM" | "LEGITIMATE" | "UNCERTAIN",
  "tiers_detected": [list of integer tier numbers present, e.g. [1, 3] or []],
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "reason": "one sentence plain-language explanation"
}\
"""

_VALID_VERDICTS = {"SCAM", "LEGITIMATE", "UNCERTAIN"}
_VALID_CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}

_P_SCAM_MAP = {
    "SCAM":       1.0,
    "UNCERTAIN":  0.5,
    "LEGITIMATE": 0.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SLMResult:
    """
    Output of GeminiSLM.analyze().

    Attributes:
        verdict        : "SCAM" | "LEGITIMATE" | "UNCERTAIN"
        tiers_detected : list of Manipulation Funnel tiers found in CALLER speech.
        confidence     : "HIGH" | "MEDIUM" | "LOW"
        reason         : one-sentence plain-language explanation.
        p_scam         : 1.0 (SCAM), 0.5 (UNCERTAIN), 0.0 (LEGITIMATE).
        inference_ms   : wall-clock time for this API call (ms).
    """
    verdict        : str
    tiers_detected : List[int]
    confidence     : str
    reason         : str
    p_scam         : float
    inference_ms   : float

    def __repr__(self) -> str:
        return (
            f"SLMResult("
            f"verdict={self.verdict!r}, "
            f"tiers={self.tiers_detected}, "
            f"confidence={self.confidence!r}, "
            f"p_scam={self.p_scam:.1f}, "
            f"ms={self.inference_ms:.0f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _uncertain(inference_ms: float, reason: str = "Insufficient context.") -> SLMResult:
    """Return a safe UNCERTAIN result — used on parse failure or API error."""
    return SLMResult(
        verdict="UNCERTAIN",
        tiers_detected=[],
        confidence="LOW",
        reason=reason,
        p_scam=0.5,
        inference_ms=inference_ms,
    )


def _parse_response(raw_text: str, inference_ms: float) -> SLMResult:
    """
    Parse the raw JSON string from Gemini into an SLMResult.
    Returns UNCERTAIN/LOW on any parse or validation failure.
    """
    # Strip accidental markdown fences (``` json ... ```)
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            ln for ln in lines if not ln.strip().startswith("```")
        ).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"[SLM] JSON parse failed: {e}. Raw: {raw_text[:200]!r}")
        return _uncertain(inference_ms, "Malformed JSON response from SLM.")

    # Validate fields
    verdict = str(data.get("verdict", "")).upper()
    if verdict not in _VALID_VERDICTS:
        logger.warning(f"[SLM] Unexpected verdict value: {verdict!r}")
        return _uncertain(inference_ms, f"Unexpected verdict: {verdict!r}")

    confidence = str(data.get("confidence", "LOW")).upper()
    if confidence not in _VALID_CONFIDENCE:
        confidence = "LOW"

    # Normalize tiers_detected — Llama 3.2 sometimes returns strings like
    # ["Tier 1", "Tier 3"] instead of integers [1, 3].
    raw_tiers = data.get("tiers_detected", [])
    if not isinstance(raw_tiers, list):
        raw_tiers = []
    normalized_tiers = []
    for t in raw_tiers:
        if isinstance(t, (int, float)) and int(t) in (1, 2, 3):
            normalized_tiers.append(int(t))
        elif isinstance(t, str):
            nums = re.findall(r"\d+", t)
            if nums and int(nums[0]) in (1, 2, 3):
                normalized_tiers.append(int(nums[0]))
    tiers = sorted(set(normalized_tiers))

    # reason — primary key; fall back to "reasons" array (Llama 3.2 variant)
    reason = str(data.get("reason", "")).strip()
    if not reason:
        reasons = data.get("reasons", [])
        if isinstance(reasons, list) and reasons:
            reason = " ".join(str(r) for r in reasons)
        elif isinstance(reasons, str):
            reason = reasons
    if not reason:
        reason = "No reason provided."

    return SLMResult(
        verdict=verdict,
        tiers_detected=tiers,
        confidence=confidence,
        reason=reason,
        p_scam=_P_SCAM_MAP[verdict],
        inference_ms=inference_ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GeminiSLM
# ─────────────────────────────────────────────────────────────────────────────

class GeminiSLM:
    """
    Gemini Flash reasoning layer — produces a structured scam verdict.

    The Manipulation Funnel system prompt is embedded at init and never
    changes. heuristic_score and classifier_score are passed as soft hints
    in the user prompt so Gemini can calibrate its confidence.

    Args:
        api_key     : Gemini API key (from config.yaml → api_keys.gemini).
        model       : Gemini model ID. Default: 'gemini-2.5-flash'.
        temperature : Sampling temperature. Default: 0.1 (structured output).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.1,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._client = None
        self._load_client(api_key)

    def _load_client(self, api_key: str) -> None:
        try:
            from google import genai
            self._client = genai.Client(api_key=api_key)
            logger.info(f"[SLM] Gemini client initialised (model: {self.model}).")
        except ImportError:
            raise ImportError(
                "google-genai is not installed. Run: pip install google-genai"
            )

    def _build_prompt(
        self,
        slm_context: str,
        heuristic_score: float,
        classifier_score: float,
    ) -> str:
        """
        Assemble the user-turn prompt.
        Prior signal scores are included as soft hints — they do NOT
        override the system prompt verdict rules.
        """
        return (
            f"Prior signals — heuristic_score: {heuristic_score:.2f}, "
            f"classifier_score: {classifier_score:.2f}\n\n"
            f"Conversation:\n{slm_context}\n\n"
            f"Respond with JSON matching this schema exactly:\n{_RESPONSE_SCHEMA}"
        )

    def analyze(
        self,
        slm_context: str,
        heuristic_score: float = 0.0,
        classifier_score: float = 0.5,
    ) -> SLMResult:
        """
        Analyze a conversation and return a structured scam verdict.

        Args:
            slm_context      : output of buf.as_slm_context() — timestamped
                               dialogue with CALLER/USER labels.
            heuristic_score  : float 0–0.5 from HeuristicDetector (soft hint).
            classifier_score : float 0–1 from MuRILClassifier (soft hint).

        Returns:
            SLMResult. On API error or malformed JSON returns UNCERTAIN/LOW —
            the pipeline never crashes on SLM failure.
        """
        if self._client is None:
            return _uncertain(0.0, "SLM client not initialised.")

        t0 = time.perf_counter()
        prompt = self._build_prompt(slm_context, heuristic_score, classifier_score)

        try:
            from google.genai import types

            # Disable thinking for gemini-2.5-* models: thinking tokens
            # consume the output budget before the JSON response begins,
            # causing truncation. For structured JSON we want direct output.
            thinking_cfg = None
            if "2.5" in self.model:
                thinking_cfg = types.ThinkingConfig(thinking_budget=0)

            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    temperature=self.temperature,
                    max_output_tokens=512,
                    thinking_config=thinking_cfg,
                ),
            )
            raw_text = response.text or ""

        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.warning(f"[SLM] API call failed: {e}")
            return _uncertain(elapsed_ms, f"API error: {type(e).__name__}")

        elapsed_ms = (time.perf_counter() - t0) * 1000
        result = _parse_response(raw_text, elapsed_ms)
        logger.info(
            f"[SLM] {result.verdict} ({result.confidence}) "
            f"tiers={result.tiers_detected} {elapsed_ms:.0f}ms"
        )
        return result

    def __repr__(self) -> str:
        return (
            f"GeminiSLM(model={self.model!r}, "
            f"temperature={self.temperature}, "
            f"connected={self._client is not None})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Ollama utility
# ─────────────────────────────────────────────────────────────────────────────

def check_ollama_running(host: str = "http://localhost:11434") -> bool:
    """Returns True if the Ollama server is reachable."""
    try:
        import httpx
        r = httpx.get(f"{host}/api/tags", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# OllamaLlamaSLM
# ─────────────────────────────────────────────────────────────────────────────

class OllamaLlamaSLM:
    """
    On-device SLM via Ollama — same verdict interface as GeminiSLM.

    Runs Llama 3.2 3B (or any Ollama-compatible model) locally.
    Requires Ollama to be running: `ollama serve`
    Pull the model first: `ollama pull llama3.2:3b`

    Shares _SYSTEM_PROMPT with GeminiSLM — same Manipulation Funnel
    guardrail, same JSON schema, same SLMResult output.

    Args:
        model       : Ollama model tag. Default: 'llama3.2:3b'.
        host        : Ollama server URL. Default: 'http://localhost:11434'.
        temperature : Sampling temperature. Default: 0.1.
    """

    def __init__(
        self,
        model: str = "llama3.2:3b",
        host: str = "http://localhost:11434",
        temperature: float = 0.1,
    ) -> None:
        self.model = model
        self.host = host
        self.temperature = temperature
        logger.info(f"[SLM] OllamaLlamaSLM initialised (model: {self.model}, host: {self.host}).")

    def analyze(
        self,
        slm_context: str,
        heuristic_score: float = 0.0,
        classifier_score: float = 0.5,
    ) -> SLMResult:
        """
        Analyze a conversation and return a structured scam verdict.

        Args:
            slm_context      : output of buf.as_slm_context().
            heuristic_score  : float 0–0.5 from HeuristicDetector (soft hint).
            classifier_score : float 0–1 from MuRILClassifier (soft hint).

        Returns:
            SLMResult. Returns UNCERTAIN/LOW on connection error or malformed
            JSON — the pipeline never crashes on SLM failure.
        """
        t0 = time.perf_counter()
        user_prompt = (
            f"Prior signals: heuristic_score={heuristic_score:.2f}, "
            f"classifier_score={classifier_score:.2f}\n\n"
            f"Conversation:\n{slm_context}\n\n"
            f"Respond with JSON only."
        )

        try:
            import ollama

            response = ollama.Client(host=self.host).chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                options={
                    "temperature": self.temperature,
                    "num_predict": 300,
                },
            )
            raw_text = response.message.content or ""

        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            # Distinguish connection errors (Ollama not running) from other errors
            err_str = str(e)
            if "connection" in err_str.lower() or "refused" in err_str.lower() or "connect" in err_str.lower():
                logger.warning(
                    f"[SLM] Ollama not reachable at {self.host}. "
                    f"Start it with: ollama serve\n"
                    f"  Then: ollama pull {self.model}"
                )
                return _uncertain(elapsed_ms, "Ollama not running — connection refused.")
            logger.warning(f"[SLM] Ollama call failed: {e}")
            return _uncertain(elapsed_ms, f"Ollama error: {type(e).__name__}")

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Strip markdown fences (Llama sometimes wraps output)
        text = raw_text.strip()
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        result = _parse_response(text, elapsed_ms)
        logger.info(
            f"[SLM] Ollama {result.verdict} ({result.confidence}) "
            f"tiers={result.tiers_detected} {elapsed_ms:.0f}ms"
        )
        return result

    def __repr__(self) -> str:
        return (
            f"OllamaLlamaSLM(model={self.model!r}, "
            f"host={self.host!r}, "
            f"temperature={self.temperature})"
        )
