"""Fail-closed semantic command boundary for an interactive AI terminal."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TerminalCommand:
    name: str
    object_id: Optional[str] = None


class SemanticTerminalPolicy:
    """Parse only high-level intents; coordinates and actuators are forbidden."""

    _READ_ONLY = {"status", "observe", "advise_grasp"}
    _MOTION = {"plan_grasp", "grasp", "release", "abort"}

    def __init__(self, *, execution_enabled=False):
        self.execution_enabled = bool(execution_enabled)

    def parse(self, text):
        words = str(text).strip().lower().split()

        if words == ["status"]:
            return TerminalCommand("status")

        if len(words) == 2 and words[0] == "observe":
            return TerminalCommand("observe", words[1])

        if words[:2] == ["plan", "grasp"] and len(words) == 3:
            return TerminalCommand("plan_grasp", words[2])

        if words[:2] == ["advise", "grasp"] and len(words) == 3:
            return TerminalCommand("advise_grasp", words[2])

        if len(words) == 2 and words[0] == "grasp":
            return TerminalCommand("grasp", words[1])

        if words == ["release"]:
            return TerminalCommand("release")

        if words == ["abort"]:
            return TerminalCommand("abort")

        raise ValueError("TERMINAL_COMMAND_UNSUPPORTED")

    def authorize(self, command):
        if command.name in self._READ_ONLY:
            return True, "READ_ONLY_COMMAND"

        if command.name == "abort":
            return True, "ABORT_ALLOWED"

        if command.name in self._MOTION and self.execution_enabled:
            return True, "SEMANTIC_EXECUTION_ALLOWED"

        return False, "SEMANTIC_EXECUTION_DISABLED"
