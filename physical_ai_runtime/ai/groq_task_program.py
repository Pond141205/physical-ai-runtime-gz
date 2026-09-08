"""Groq-backed natural-language interpreter for validated task-space programs."""

import json
import os
from pathlib import Path

from groq import Groq

from physical_ai_runtime.ai.semantic_manipulation_task import SemanticTaskProgram


class GroqTaskProgramInterpreter:
    def __init__(self, model="openai/gpt-oss-120b", client=None):
        if client is None:
            self._load_user_config()
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY_NOT_SET")
            client = Groq(api_key=api_key)
        self.client = client
        self.model = model

    @staticmethod
    def _load_user_config():
        """Load only the Groq key from a user-owned, non-repository file."""
        config_root = Path(
            os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
        )
        config_path = config_root / "physical-ai-runtime" / "groq.env"

        if os.environ.get("GROQ_API_KEY") or not config_path.is_file():
            return

        for line in config_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("GROQ_API_KEY="):
                value = line.partition("=")[2].strip()
                if value:
                    os.environ["GROQ_API_KEY"] = value
                return

    def interpret(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Convert the user request into JSON {steps:[...]}. "
                        "Allowed executable actions: MOVE_TO, OPEN, CLOSE, STOP. "
                        "MOVE_TO requires frame_id and position [x,y,z], optional orientation [x,y,z,w]. "
                        "OPEN and CLOSE operate the configured gripper. STOP stops motion. "
                        "Do not emit PICK, PICK_AND_PLACE, GRASP, RELEASE, HOLD, or MOVE_POSE. "
                        "Never output joints, torques, controller commands, trajectories, velocities, or robot names."
                    ),
                },
                {"role": "user", "content": str(prompt)},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("GROQ_EMPTY_TASK_PROGRAM")
        return SemanticTaskProgram.from_model_output(json.loads(content))
