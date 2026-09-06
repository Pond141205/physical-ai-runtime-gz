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

Allowed semantic actions:
- PICK
- PICK_AND_PLACE
- HOLD
- MOVE_POSE
- GRASP
- RELEASE

PICK:
{
  "action": "PICK",
  "object_id": "cube"
}

GRASP:
{
  "action": "GRASP",
  "object_id": "cube"
}

RELEASE:
{
  "action": "RELEASE"
}

PICK_AND_PLACE:
{
  "action": "PICK_AND_PLACE",
  "object_id": "cube",
  "relation": "beside",
  "reference_object_id": "cylinder"
}

MOVE_POSE is allowed only when the user explicitly supplies or clearly
requests a task-space pose:
{
  "action": "MOVE_POSE",
  "frame_id": "world",
  "position": [x, y, z],
  "orientation": [x, y, z, w]
}

Never invent metric coordinates for an object from conversation.
Object geometry must come from perception/runtime, not language guessing.

Respond in the user's language.
""".strip()

    def respond(self, text):
        user_text = str(text).strip()

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
                "content": user_text,
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=messages,
        )

        content = response.choices[0].message.content

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
