"""Scripted multi-turn evaluation: turn accuracy and conversation (task) success rate."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sherlock_rx.dialogue.manager import Sherlock

ROOT = Path(__file__).resolve().parents[3]
DIALOGUES = ROOT / "tests" / "data" / "dialogues.yaml"


@dataclass
class DialogueResult:
    conversations: int
    success_rate: float
    turns: int
    turn_accuracy: float
    p50_turn_ms: float
    p95_turn_ms: float
    failures: list[dict[str, Any]] = field(default_factory=list)


def check_turn(turn: dict[str, Any], resp: Any) -> list[str]:
    errs = []
    if "intent" in turn and resp.intent != turn["intent"]:
        errs.append(f"intent {resp.intent} != {turn['intent']}")
    if "policy" in turn and turn["policy"] not in resp.policy:
        errs.append(f"policy '{resp.policy}' lacks '{turn['policy']}'")
    for k, v in (turn.get("state") or {}).items():
        got = resp.state.get(k)
        if got != v:
            errs.append(f"state.{k} {got!r} != {v!r}")
    for sub in turn.get("contains", []):
        if sub.lower() not in resp.text.lower():
            errs.append(f"reply lacks '{sub}'")
    return errs


def evaluate(bot: Sherlock, path: Path = DIALOGUES) -> DialogueResult:
    convs = yaml.safe_load(path.read_text())
    ok_convs = ok_turns = n_turns = 0
    lat: list[float] = []
    failures = []
    for conv in convs:
        state = bot.new_state()
        conv_ok = True
        for i, turn in enumerate(conv["turns"]):
            resp = bot.handle(turn["say"], state)
            lat.append(resp.timings_ms.get("total", 0.0))
            errs = check_turn(turn, resp)
            n_turns += 1
            if errs:
                conv_ok = False
                failures.append({"conversation": conv["name"], "turn": i + 1,
                                 "say": turn["say"], "errors": errs})
            else:
                ok_turns += 1
        ok_convs += conv_ok
    lat.sort()
    return DialogueResult(len(convs), ok_convs / len(convs), n_turns, ok_turns / n_turns,
                          lat[len(lat) // 2], lat[int(len(lat) * 0.95)], failures)
