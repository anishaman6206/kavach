"""
kavach.audio.buffer
====================
Rolling conversation buffer — the single source of truth shared by
every downstream layer (heuristics, MuRIL, SLM).

Design decisions:
  - Stores last N utterances with CALLER / USER speaker tags
  - Never flushed mid-call; oldest utterance falls off automatically
  - Provides formatted context strings ready for MuRIL and SLM input
  - Pure Python / dataclasses — no ML dependencies, fully testable
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ─────────────────────────────────────────────
# Shared types (imported by vad.py, transcription, detection)
# ─────────────────────────────────────────────

class Speaker(Enum):
    """Who is speaking in a given utterance."""
    CALLER = "CALLER"   # the other person on the call — scam signals apply here
    USER   = "USER"     # the phone owner — never flagged for Tier signals


@dataclass
class Utterance:
    """
    One turn of speech, tagged with speaker and timing.

    Attributes:
        speaker      : CALLER or USER
        text         : transcribed text (lowercased, cleaned)
        timestamp    : seconds from call start
        chunk_index  : which 2.5s audio chunk this came from
        raw_text     : original text before cleaning (kept for debugging)
        language     : ISO 639-1 code detected by Whisper ('hi', 'en', 'te' …)
    """
    speaker     : Speaker
    text        : str
    timestamp   : float
    chunk_index : int
    raw_text    : str  = ""
    language    : str  = "en"

    def __post_init__(self) -> None:
        # Keep raw_text for debugging; clean text for downstream
        if not self.raw_text:
            self.raw_text = self.text
        self.text = self._clean(self.text)

    @staticmethod
    def _clean(text: str) -> str:
        """Lowercase and strip leading/trailing whitespace."""
        return text.strip().lower()

    def __repr__(self) -> str:
        preview = self.text[:60] + "..." if len(self.text) > 60 else self.text
        return f"Utterance({self.speaker.value} @{self.timestamp:.1f}s: '{preview}')"


# ─────────────────────────────────────────────
# Rolling buffer
# ─────────────────────────────────────────────

class ConversationBuffer:
    """
    Maintains the last `max_utterances` turns of a call, speaker-tagged.

    Usage:
        buf = ConversationBuffer(max_utterances=10)
        buf.add(Utterance(Speaker.CALLER, "I am calling from SBI", 0.0, 0))
        buf.add(Utterance(Speaker.USER,   "Yes, who is this?",     3.2, 1))

        # For MuRIL: full context as one string
        text = buf.as_classifier_input()

        # For SLM: structured dialogue string
        dialogue = buf.as_slm_context()

        # For heuristics: only what the caller said
        caller_text = buf.caller_text_only()
    """

    def __init__(
        self,
        max_utterances: int = 10,
        max_tokens_approx: int = 600,
    ) -> None:
        """
        Args:
            max_utterances    : hard cap on number of utterances kept.
                                Oldest drops off automatically (deque).
            max_tokens_approx : soft cap — context strings are truncated
                                at roughly this many tokens (1 token ≈ 4 chars).
        """
        if max_utterances < 1:
            raise ValueError("max_utterances must be at least 1")

        self._buffer            : deque[Utterance] = deque(maxlen=max_utterances)
        self.max_utterances     : int = max_utterances
        self._max_chars         : int = max_tokens_approx * 4  # rough char budget

    # ── Core operations ──────────────────────────────────────────────────

    def add(self, utterance: Utterance) -> None:
        """
        Append a new utterance. If buffer is full, the oldest is evicted
        automatically (deque behaviour).
        """
        self._buffer.append(utterance)

    def clear(self) -> None:
        """Wipe the buffer — call at end of call session."""
        self._buffer.clear()

    # ── Read operations ──────────────────────────────────────────────────

    @property
    def utterances(self) -> List[Utterance]:
        """All utterances in chronological order."""
        return list(self._buffer)

    @property
    def is_empty(self) -> bool:
        return len(self._buffer) == 0

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def latest(self) -> Optional[Utterance]:
        """Most recent utterance, or None if buffer is empty."""
        return self._buffer[-1] if self._buffer else None

    def caller_utterances(self) -> List[Utterance]:
        """Only CALLER turns — used by heuristic tier detector."""
        return [u for u in self._buffer if u.speaker == Speaker.CALLER]

    def user_utterances(self) -> List[Utterance]:
        """Only USER turns — used for response context."""
        return [u for u in self._buffer if u.speaker == Speaker.USER]

    # ── Formatted outputs ────────────────────────────────────────────────

    def caller_text_only(self) -> str:
        """
        Concatenated CALLER speech as a single string.
        Fed to the heuristic tier detector (regex patterns run here).

        Example:
            "i am calling from sbi your account shows suspicious activity
             please share your otp to verify"
        """
        return " ".join(u.text for u in self.caller_utterances())

    def as_classifier_input(self) -> str:
        """
        Full conversation as a single flat string for MuRIL.
        Speaker labels included so the model learns turn structure.
        Truncated from the left if it exceeds the character budget
        (most recent utterances are most important).

        Example:
            "CALLER: i am calling from sbi USER: yes who is this
             CALLER: your account is suspended please share otp"
        """
        parts = [f"{u.speaker.value}: {u.text}" for u in self._buffer]
        full  = " ".join(parts)

        # Truncate from left (keep recent context)
        if len(full) > self._max_chars:
            full = full[-self._max_chars:]
            # Don't start mid-word
            first_space = full.find(" ")
            if first_space != -1:
                full = full[first_space + 1:]

        return full

    def as_slm_context(self) -> str:
        """
        Structured dialogue string for the SLM (Gemini / Llama).
        Each turn on its own line with speaker label and timestamp.
        This is what gets injected into the SLM prompt.

        Example:
            [0.0s] CALLER: hello i am calling from sbi bank
            [3.2s] USER: yes who is this
            [5.8s] CALLER: your account shows suspicious activity
        """
        lines = [
            f"[{u.timestamp:.1f}s] {u.speaker.value}: {u.text}"
            for u in self._buffer
        ]
        dialogue = "\n".join(lines)

        # Truncate from top if too long (keep recent)
        if len(dialogue) > self._max_chars:
            lines_trunc = []
            chars = 0
            for line in reversed(lines):
                chars += len(line) + 1
                if chars > self._max_chars:
                    break
                lines_trunc.insert(0, line)
            dialogue = "\n".join(lines_trunc)

        return dialogue

    def detected_languages(self) -> List[str]:
        """Unique languages detected across all utterances in the buffer."""
        return list(dict.fromkeys(u.language for u in self._buffer))

    # ── Debug / display ──────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"ConversationBuffer("
            f"size={self.size}/{self.max_utterances}, "
            f"latest={self.latest})"
        )

    def pretty_print(self) -> None:
        """Print the buffer contents for debugging."""
        print(f"\n── Conversation Buffer ({self.size} utterances) ──")
        for u in self._buffer:
            print(f"  [{u.timestamp:6.1f}s] {u.speaker.value:6s}: {u.text}")
        print("──────────────────────────────────────────────\n")