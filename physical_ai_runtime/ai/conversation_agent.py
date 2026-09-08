"""Conversational front-end for the Physical AI runtime.

The agent may converse freely and propose semantic task programs.
It never owns motion authority.
"""

from dataclasses import dataclass
import json
import os

from groq import Groq

from physical_ai_runtime.ai.groq_task_program import (
    GroqTaskProgramInterpreter,
)
from physical_ai_runtime.ai.semantic_manipulation_task import (
    SemanticTaskProgram,
)


@dataclass(frozen=True)
class ConversationTurn:
    mode: str
    message: str
    program: SemanticTaskProgram | None = None


def _utf8_safe(value):
    """Remove malformed surrogate code points from model/runtime text."""
    if isinstance(value, str):
        return value.encode(
            "utf-8",
            errors="replace",
        ).decode("utf-8")

    if isinstance(value, dict):
        return {
            _utf8_safe(k): _utf8_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            _utf8_safe(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            _utf8_safe(item)
            for item in value
        )

    return value


class PhysicalAIConversationAgent:
    """Natural-language boundary in front of RobotRuntime."""

    def __init__(
        self,
        model="openai/gpt-oss-120b",
        client=None,
        history_limit=12,
    ):
        if client is None:
            GroqTaskProgramInterpreter._load_user_config()

            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY_NOT_SET")

            client = Groq(api_key=api_key)

        self.client = client
        self.model = model
        self.history_limit = int(history_limit)
        self.history = []

    def _system_prompt(self):
        return """
You are the conversational interface of a safety-first Physical AI runtime.

You can:
- talk naturally with the user
- answer questions about your role and capabilities
- discuss the robot and task at a semantic level
- interpret a user request as a semantic robot task

You are NOT the motion authority.

Architecture:
User
-> Conversational AI
-> validated semantic task
-> RobotRuntime
-> safety verification
-> MoveIt planning
-> execution authorization
-> robot adapter/controller

You MUST NOT:
- directly command joints
- directly command motors
- directly command torque
- directly command velocity
- directly generate controller commands
- directly authorize robot motion
- claim that motion happened before RobotRuntime reports success
- bypass TF, workspace, PlanningScene, capability, planning, or authorization checks

Return one JSON object.

For ordinary conversation:
{
  "mode": "chat",
  "message": "natural response to the user"
}

For a robot action request:
{
  "mode": "task",
  "message": "short natural acknowledgement",
  "program": {
    "steps": [
      {
        "action": "..."
      }
    ]
  }
}

Currently executable primitive actions:
- MOVE_TO: an explicitly supplied task-space pose
- OPEN: open the configured gripper
- CLOSE: close the configured gripper
- STOP: stop current motion

Never emit PICK, PICK_AND_PLACE, GRASP, RELEASE, HOLD, or MOVE_POSE.

MOVE_TO is allowed only when the user explicitly supplies or clearly
requests a task-space pose:
{
  "action": "MOVE_TO",
  "frame_id": "world",
  "position": [x, y, z],
  "orientation": [x, y, z, w]
}

Never invent metric coordinates for an object from conversation.
Object geometry must come from perception/runtime, not language guessing.

Respond in the user's language.
        """.strip()

    def record_runtime_feedback(self, feedback):
        """Expose authoritative runtime feedback to the next AI turn."""
        self.history.append({
            "role": "system",
            "content": (
                "AUTHORITATIVE RUNTIME FEEDBACK (not a user request):\n"
                + json.dumps(
                    _utf8_safe(feedback),
                    ensure_ascii=False,
                    default=str,
                )
                + "\nUse this for state awareness; do not claim success "
                  "unless it reports success."
            ),
        })
        self.history = self.history[-(self.history_limit * 2):]

    def respond(self, text, runtime_context=None):
        user_text = _utf8_safe(
            str(text).strip()
        )

        runtime_context = _utf8_safe(
            runtime_context
        )

        if not user_text:
            raise ValueError("CONVERSATION_EMPTY_INPUT")

        messages = [
            {
                "role": "system",
                "content": self._system_prompt(),
            },
            *self.history[-self.history_limit:],
            {
                "role": "user",
                "content": (
                    user_text
                    if runtime_context is None
                    else (
                        user_text
                        + "\n\nLIVE VISUAL CONTEXT:\n"
                        + json.dumps(
                            runtime_context,
                            ensure_ascii=False,
                            default=str,
                        )
                        + "\nUse this as current visual evidence. "
                          "Do not invent details not present."
                    )
                ),
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=messages,
        )

        content = _utf8_safe(
            response.choices[0].message.content
        )

        if not content:
            raise RuntimeError("GROQ_EMPTY_CONVERSATION_RESPONSE")

        value = json.loads(content)

        if not isinstance(value, dict):
            raise RuntimeError("CONVERSATION_INVALID_RESPONSE")

        mode = str(value.get("mode", "")).strip().lower()
        message = str(value.get("message", "")).strip()

        if mode not in {"chat", "task"}:
            raise RuntimeError(
                "CONVERSATION_INVALID_MODE:" + mode
            )

        if not message:
            raise RuntimeError(
                "CONVERSATION_EMPTY_MESSAGE"
            )

        program = None

        if mode == "task":
            program = SemanticTaskProgram.from_model_output(
                value.get("program")
            )

        self.history.extend(
            [
                {
                    "role": "user",
                    "content": user_text,
                },
                {
                    "role": "assistant",
                    "content": content,
                },
            ]
        )

        self.history = self.history[
            -(self.history_limit * 2):
        ]

        return ConversationTurn(
            mode=mode,
            message=message,
            program=program,
        )
