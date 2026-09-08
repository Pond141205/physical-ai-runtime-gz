import unittest
import os
import tempfile
from pathlib import Path

from physical_ai_runtime.ai.groq_task_program import GroqTaskProgramInterpreter


class _Completions:
    def create(self, **_kwargs):
        message = type("Message", (), {
            "content": '{"steps":[{"action":"MOVE_TO","frame_id":"world","position":[0.4,0.1,0.5]}]}'
        })()
        return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()


class _Client:
    chat = type("Chat", (), {"completions": _Completions()})()


class GroqTaskProgramInterpreterTest(unittest.TestCase):
    def test_interpreter_validates_model_output_before_runtime(self):
        program = GroqTaskProgramInterpreter(client=_Client()).interpret("wave")

        self.assertEqual(program.steps[0].action, "MOVE_TO")
        self.assertEqual(program.steps[0].frame_id, "world")

    def test_loads_key_from_user_config_without_repository_file(self):
        old_key = os.environ.pop("GROQ_API_KEY", None)
        old_config = os.environ.get("XDG_CONFIG_HOME")
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.environ["XDG_CONFIG_HOME"] = directory
                config = Path(directory) / "physical-ai-runtime" / "groq.env"
                config.parent.mkdir()
                config.write_text("GROQ_API_KEY=test-key\n", encoding="utf-8")

                GroqTaskProgramInterpreter._load_user_config()

                self.assertEqual(os.environ["GROQ_API_KEY"], "test-key")
        finally:
            os.environ.pop("GROQ_API_KEY", None)
            if old_key:
                os.environ["GROQ_API_KEY"] = old_key
            if old_config is None:
                os.environ.pop("XDG_CONFIG_HOME", None)
            else:
                os.environ["XDG_CONFIG_HOME"] = old_config


if __name__ == "__main__":
    unittest.main()
