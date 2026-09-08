import unittest

from physical_ai_runtime.ai.semantic_terminal import SemanticTerminalPolicy


class SemanticTerminalPolicyTest(unittest.TestCase):
    def test_read_only_commands_are_allowed(self):
        policy = SemanticTerminalPolicy()

        command = policy.parse("observe cube")

        self.assertEqual(command.name, "observe")
        self.assertEqual(command.object_id, "cube")
        self.assertEqual(policy.authorize(command), (True, "READ_ONLY_COMMAND"))

        task = policy.parse("task move to the requested pose")
        self.assertEqual(task.name, "task_program")
        self.assertEqual(task.prompt, "move to the requested pose")
        self.assertEqual(policy.authorize(task), (True, "READ_ONLY_COMMAND"))

    def test_motion_is_disabled_by_default(self):
        policy = SemanticTerminalPolicy()

        command = policy.parse("open")

        self.assertEqual(
            policy.authorize(command),
            (False, "SEMANTIC_EXECUTION_DISABLED"),
        )

    def test_move_to_accepts_task_space_coordinates(self):
        policy = SemanticTerminalPolicy(execution_enabled=True)
        command = policy.parse("move to world 0.4 0.1 0.5")

        self.assertEqual(command.name, "move_to")
        self.assertEqual(command.frame_id, "world")
        self.assertEqual(command.position, (0.4, 0.1, 0.5))
        self.assertEqual(
            policy.authorize(command),
            (True, "PRIMITIVE_EXECUTION_ALLOWED"),
        )

    def test_rejects_object_and_joint_commands(self):
        policy = SemanticTerminalPolicy(execution_enabled=True)

        for text in (
            "joint 0 0 0 0 0 0 0",
            "grasp cube",
            "release",
            "pick cube",
        ):
            with self.assertRaisesRegex(
                ValueError,
                "TERMINAL_COMMAND_UNSUPPORTED",
            ):
                policy.parse(text)

    def test_explicit_execution_only_allows_primitives(self):
        policy = SemanticTerminalPolicy(execution_enabled=True)

        command = policy.parse("close")

        self.assertEqual(
            policy.authorize(command),
            (True, "PRIMITIVE_EXECUTION_ALLOWED"),
        )

    def test_stop_is_always_allowed(self):
        policy = SemanticTerminalPolicy()
        self.assertEqual(
            policy.authorize(policy.parse("stop")),
            (True, "STOP_ALLOWED"),
        )


if __name__ == "__main__":
    unittest.main()
