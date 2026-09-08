"""Fail-closed semantic command boundary for an interactive AI terminal."""

from dataclasses import dataclass
import math
from typing import Optional


@dataclass(frozen=True)
class TerminalCommand:
    name: str
    object_id: Optional[str] = None
    prompt: Optional[str] = None
    frame_id: Optional[str] = None
    position: Optional[tuple[float, float, float]] = None
    orientation: Optional[tuple[float, float, float, float]] = None


class SemanticTerminalPolicy:
    """Parse a small task-space primitive command surface."""

    _READ_ONLY = {"status", "observe", "task_program"}
    _MOTION = {"move_to", "open", "close"}

    def __init__(self, *, execution_enabled=False):
        self.execution_enabled = bool(execution_enabled)

    def parse(self, text):
        raw = str(text).strip()
        if raw.lower().startswith("task "):
            prompt = raw[5:].strip()
            if not prompt:
                raise ValueError("TERMINAL_TASK_PROMPT_REQUIRED")
            return TerminalCommand("task_program", prompt=prompt)

        words = raw.lower().split()

        if words == ["status"]:
            return TerminalCommand("status")

        if len(words) == 2 and words[0] == "observe":
            return TerminalCommand("observe", words[1])

        if words in (["open"], ["close"]):
            return TerminalCommand(words[0])

        if words in (["stop"], ["abort"]):
            return TerminalCommand("stop")

        if words[:2] == ["move", "to"] and len(words) in {6, 10}:
            try:
                values = tuple(float(value) for value in words[3:])
            except ValueError as exc:
                raise ValueError("TERMINAL_COMMAND_INVALID_POSE") from exc

            if not all(math.isfinite(value) for value in values):
                raise ValueError("TERMINAL_COMMAND_INVALID_POSE")

            return TerminalCommand(
                "move_to",
                frame_id=words[2],
                position=values[:3],
                orientation=(None if len(values) == 3 else values[3:]),
            )

        raise ValueError("TERMINAL_COMMAND_UNSUPPORTED")

    def authorize(self, command):
        if command.name in self._READ_ONLY:
            return True, "READ_ONLY_COMMAND"

        if command.name == "stop":
            return True, "STOP_ALLOWED"

        if command.name in self._MOTION and self.execution_enabled:
            return True, "PRIMITIVE_EXECUTION_ALLOWED"

        return False, "SEMANTIC_EXECUTION_DISABLED"
