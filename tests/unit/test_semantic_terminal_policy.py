import unittest

from physical_ai_runtime.ai.semantic_terminal import SemanticTerminalPolicy


class SemanticTerminalPolicyTest(unittest.TestCase):
    def test_read_only_commands_are_allowed(self):
        policy = SemanticTerminalPolicy()

        command = policy.parse("observe cube")

        self.assertEqual(command.name, "observe")
        self.assertEqual(command.object_id, "cube")
        self.assertEqual(policy.authorize(command), (True, "READ_ONLY_COMMAND"))

        advice = policy.parse("advise grasp cube")
        self.assertEqual(advice.name, "advise_grasp")
        self.assertEqual(advice.object_id, "cube")
        self.assertEqual(policy.authorize(advice), (True, "READ_ONLY_COMMAND"))

    def test_motion_is_disabled_by_default(self):
        policy = SemanticTerminalPolicy()

        command = policy.parse("grasp cube")

        self.assertEqual(
            policy.authorize(command),
            (False, "SEMANTIC_EXECUTION_DISABLED"),
        )

    def test_motion_never_accepts_coordinates_or_joint_commands(self):
        policy = SemanticTerminalPolicy(execution_enabled=True)

        for text in (
            "move tcp 0.4 0.1 0.5",
            "joint 0 0 0 0 0 0 0",
            "grasp cube 0.4",
        ):
            with self.assertRaisesRegex(
                ValueError,
                "TERMINAL_COMMAND_UNSUPPORTED",
            ):
                policy.parse(text)

    def test_explicit_execution_only_allows_semantic_intent(self):
        policy = SemanticTerminalPolicy(execution_enabled=True)

        command = policy.parse("grasp cube")

        self.assertEqual(
            policy.authorize(command),
            (True, "SEMANTIC_EXECUTION_ALLOWED"),
        )


if __name__ == "__main__":
    unittest.main()
