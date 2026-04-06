"""
kavach.ui.terminal_ui
======================
Rich terminal dashboard for real-time scam detection display.

Four-panel layout:
  ┌─ header: call status + current alert level ──────────────────┐
  ├─ risk score panel ──┬─ detection signals panel ──────────────┤
  ├─ conversation ──────┴────────────────────────────────────────┤
  ├─ SLM explanation ────────────────────────────────────────────┤
  └─ timeline (scrolling log) ───────────────────────────────────┘

Usage:
    ui = KavachTerminalUI()
    ui.start()
    for each risk_result, buffer:
        ui.update(risk_result, buffer)
    ui.stop()
"""

from __future__ import annotations

from typing import List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live   import Live
from rich.panel  import Panel
from rich.table  import Table
from rich.text   import Text
from rich.rule   import Rule
from rich import box

from kavach.audio.buffer      import ConversationBuffer, Speaker
from kavach.fusion.risk_scorer import RiskResult

# ── Constants ────────────────────────────────────────────────────────────────

_BAR_WIDTH   = 20
_MAX_CONV    = 5
_MAX_TIMELINE = 10

_LEVEL_COLOR = {
    "SAFE":     "green",
    "CAUTION":  "yellow",
    "ALERT":    "bold red",
    "CRITICAL": "bold red",
}

_LEVEL_ICON = {
    "SAFE":     "🟢",
    "CAUTION":  "🟡",
    "ALERT":    "🔴",
    "CRITICAL": "🚨",
}


# ── Pure helper functions (testable without a Live instance) ─────────────────

def _make_risk_bar(score: float, width: int = _BAR_WIDTH) -> str:
    """
    Build a fixed-width progress bar from filled and empty block chars.

    >>> _make_risk_bar(0.0)   == '░' * 20
    True
    >>> _make_risk_bar(1.0)   == '█' * 20
    True
    >>> _make_risk_bar(0.5)   == '█' * 10 + '░' * 10
    True
    """
    filled = round(max(0.0, min(1.0, score)) * width)
    return "█" * filled + "░" * (width - filled)


def _alert_color(level: str) -> str:
    """Return rich color string for an alert level."""
    return _LEVEL_COLOR.get(level, "white")


def _format_utterance(speaker: Speaker, text: str, timestamp: float,
                      max_chars: int = 60) -> Text:
    """
    Format a single utterance line for the conversation panel.
    CALLER → yellow, USER → blue. Text truncated at max_chars.
    """
    color  = "yellow" if speaker == Speaker.CALLER else "blue"
    label  = speaker.value
    preview = text if len(text) <= max_chars else text[:max_chars - 1] + "…"
    line = Text()
    line.append(f"[{timestamp:5.1f}s] ", style="dim")
    line.append(f"{label:6s}: ", style=f"bold {color}")
    line.append(preview, style=color)
    return line


def _tier_row(tier: int, seen: bool, label: str) -> Text:
    """Single tier status line for the signals panel."""
    line = Text()
    line.append(f"  Tier {tier}  ", style="bold")
    if seen:
        line.append("✓  ", style="bold green")
        line.append(label, style="green")
    else:
        line.append("—  ", style="dim")
        line.append(label, style="dim")
    return line


# ── Main class ───────────────────────────────────────────────────────────────

class KavachTerminalUI:
    """
    Rich Live terminal dashboard.

    Lifecycle:
        ui = KavachTerminalUI()
        ui.start()
        ui.update(risk, buf)   # call after every RiskScorer.update()
        ui.stop()
    """

    def __init__(self, refresh_per_second: float = 2.0) -> None:
        self._console   = Console()
        self._refresh   = refresh_per_second
        self._live: Optional[Live] = None
        self._layout    = Layout()
        self._timeline  : List[str] = []
        self._last_risk : Optional[RiskResult] = None
        self._elapsed_s : float = 0.0
        self._build_layout()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Begin the Live display."""
        self._live = Live(
            self._layout,
            console=self._console,
            refresh_per_second=self._refresh,
            screen=True,
        )
        self._live.start()

    def stop(self) -> None:
        """End the Live display and print a final summary."""
        if self._live:
            self._live.stop()
            self._live = None
        self._print_final_summary()

    # ── Update ────────────────────────────────────────────────────────────────

    def update(
        self,
        risk: RiskResult,
        buf: ConversationBuffer,
        elapsed_s: float = 0.0,
    ) -> None:
        """
        Refresh all panels with the latest RiskResult and buffer state.
        Safe to call even if start() hasn't been called (no-op in that case).
        """
        self._last_risk  = risk
        self._elapsed_s  = elapsed_s
        self._add_timeline_entry(risk, elapsed_s)
        self._layout["header"].update(self._render_header(risk))
        self._layout["risk"].update(self._render_risk(risk))
        self._layout["signals"].update(self._render_signals(risk))
        self._layout["conversation"].update(self._render_conversation(buf))
        self._layout["explanation"].update(self._render_explanation(risk))
        self._layout["timeline"].update(self._render_timeline())

    def show_alert(self, risk: RiskResult) -> None:
        """Flash an alert banner for ALERT/CRITICAL levels (no-op for SAFE/CAUTION)."""
        if risk.alert_level not in ("ALERT", "CRITICAL"):
            return
        color = _alert_color(risk.alert_level)
        icon  = _LEVEL_ICON[risk.alert_level]
        self._layout["header"].update(
            Panel(
                Text(
                    f"{icon}  {risk.alert_level} — {risk.explanation[:80]}",
                    style=f"bold {color}",
                    justify="center",
                ),
                style=color,
            )
        )

    # ── Layout builder ────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        self._layout.split_column(
            Layout(name="header",      size=3),
            Layout(name="middle",      size=11),
            Layout(name="conversation",size=9),
            Layout(name="explanation", size=4),
            Layout(name="timeline",    size=14),
        )
        self._layout["middle"].split_row(
            Layout(name="risk",    ratio=1),
            Layout(name="signals", ratio=1),
        )
        # Seed with empty panels so Live never renders a blank screen
        self._layout["header"].update(
            Panel(Text("KAVACH  —  Real-Time Scam Call Detection", justify="center"),
                  style="bold")
        )
        self._layout["risk"].update(Panel("Initialising…", title="RISK SCORE"))
        self._layout["signals"].update(Panel("Initialising…", title="DETECTION SIGNALS"))
        self._layout["conversation"].update(Panel("", title="CONVERSATION  (last 5 utterances)"))
        self._layout["explanation"].update(Panel("Waiting for first analysis…", title="SLM EXPLANATION"))
        self._layout["timeline"].update(Panel("", title="TIMELINE"))

    # ── Panel renderers ───────────────────────────────────────────────────────

    def _render_header(self, risk: RiskResult) -> Panel:
        color = _alert_color(risk.alert_level)
        icon  = _LEVEL_ICON[risk.alert_level]
        title = Text()
        title.append("KAVACH  —  Real-Time Scam Call Detection          ", style="bold")
        title.append(f"{icon} {risk.alert_level}", style=f"bold {color}")
        return Panel(title, style=color if risk.alert_level != "SAFE" else "")

    def _render_risk(self, risk: RiskResult) -> Panel:
        color = _alert_color(risk.alert_level)
        score = risk.final_score
        bar   = _make_risk_bar(score)

        t = Text()
        t.append("\n")
        t.append(f"  {bar}  ", style=color)
        t.append(f"{score:.2f}\n\n", style=f"bold {color}")
        t.append(f"  {risk.alert_level}\n\n", style=f"bold {color}")
        t.append(f"  Heuristic:   {risk.component_scores.get('heuristic', 0):.2f}\n", style="dim")
        t.append(f"  Classifier:  {risk.component_scores.get('classifier', 0):.2f}\n", style="dim")
        t.append(f"  SLM:         {risk.component_scores.get('slm', 0):.2f}\n",        style="dim")

        return Panel(t, title="[bold]RISK SCORE[/bold]", box=box.ROUNDED)

    def _render_signals(self, risk: RiskResult) -> Panel:
        tiers = set(risk.tiers_seen_this_call)
        labels = {
            1: "Authority claim",
            2: "Urgency / threat",
            3: "Credential request",
        }
        t = Text()
        t.append("\n")
        for tier_num in (1, 2, 3):
            t.append_text(_tier_row(tier_num, tier_num in tiers, labels[tier_num]))
            t.append("\n")
        t.append("\n")
        t.append(f"  Heuristic:   {risk.component_scores.get('heuristic', 0):.2f}\n", style="dim")
        t.append(f"  Classifier:  {risk.component_scores.get('classifier', 0):.2f}\n", style="dim")
        t.append(f"  SLM:         {risk.component_scores.get('slm', 0):.2f}\n",        style="dim")

        return Panel(t, title="[bold]DETECTION SIGNALS[/bold]", box=box.ROUNDED)

    def _render_conversation(self, buf: ConversationBuffer) -> Panel:
        utterances = buf.utterances[-_MAX_CONV:]
        t = Text()
        t.append("\n")
        if not utterances:
            t.append("  (no speech yet…)", style="dim")
        else:
            for utt in utterances:
                t.append_text(_format_utterance(utt.speaker, utt.raw_text or utt.text, utt.timestamp))
                t.append("\n")
        return Panel(t, title="[bold]CONVERSATION[/bold]  (last 5 utterances)", box=box.ROUNDED)

    def _render_explanation(self, risk: RiskResult) -> Panel:
        explanation = risk.explanation or "No analysis yet."
        t = Text(f'\n  "{explanation}"', style="italic")
        return Panel(t, title="[bold]SLM EXPLANATION[/bold]", box=box.ROUNDED)

    def _render_timeline(self) -> Panel:
        t = Text()
        t.append("\n")
        for entry in self._timeline[-_MAX_TIMELINE:]:
            t.append(entry + "\n")
        return Panel(t, title="[bold]TIMELINE[/bold]", box=box.ROUNDED)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _add_timeline_entry(self, risk: RiskResult, elapsed_s: float) -> None:
        color = _alert_color(risk.alert_level)
        tiers = "+".join(str(t) for t in risk.tiers_seen_this_call) or "—"
        expl  = risk.explanation[:40] + "…" if len(risk.explanation) > 40 else risk.explanation
        bar_short = "█" * round(risk.final_score * 5)   # 5-char mini bar
        entry = (
            f"  [dim]{elapsed_s:4.0f}s[/dim]  "
            f"[{color}]{risk.alert_level:<8}[/{color}]  "
            f"[{color}]{risk.final_score:.2f}[/{color}]  "
            f"[{color}]{bar_short:<5}[/{color}]  "
            f"[dim]Tiers[{tiers}][/dim]  {expl}"
        )
        self._timeline.append(entry)

    def _print_final_summary(self) -> None:
        if self._last_risk is None:
            self._console.print("\n[dim]No pipeline results recorded.[/dim]")
            return
        r = self._last_risk
        color = _alert_color(r.alert_level)
        self._console.print()
        self._console.rule("[bold]KAVACH — Final Summary[/bold]")
        self._console.print(f"  Verdict      : [{color}]{r.alert_level}[/{color}]")
        self._console.print(f"  Peak score   : [{color}]{r.final_score:.3f}[/{color}]")
        self._console.print(f"  Tiers seen   : {r.tiers_seen_this_call or '(none)'}")
        self._console.print(f"  Explanation  : {r.explanation}")
        self._console.print()
