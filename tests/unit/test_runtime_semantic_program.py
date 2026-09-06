import unittest

import numpy as np

from physical_ai_runtime.ai.semantic_manipulation_task import SemanticTaskProgram
from physical_ai_runtime.core.runtime import RobotRuntime


class _FakeRobot:
    base_frame = "test_base"

    def __init__(self):
        self.calls = []

    def get_tcp_pose(self):
        return np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])

    def plan_tcp_pose(self, *, target, orientation):
        self.calls.append((np.asarray(target), np.asarray(orientation)))
        return {
            "success": True,
            "trajectory": object(),
            "planned_at_monotonic": 1.0,
        }


class RuntimeSemanticProgramTest(unittest.TestCase):
    def test_plans_all_gesture_steps_in_adapter_base_frame(self):
        program = SemanticTaskProgram.from_model_output({
            "steps": [
                {"action": "MOVE_POSE", "frame_id": "test_base", "position": [0.1, 0.2, 0.3]},
                {"action": "MOVE_POSE", "frame_id": "test_base", "position": [0.1, -0.2, 0.3]},
            ]
        })
        robot = _FakeRobot()

        result = RobotRuntime().execute_semantic_program(robot, program, execute=False)

        self.assertTrue(result.success)
        self.assertEqual(len(robot.calls), 2)
        np.testing.assert_allclose(robot.calls[1][0], [0.1, -0.2, 0.3])

    def test_missing_non_base_frame_fails_closed(self):
        program = SemanticTaskProgram.from_model_output({
            "steps": [
                {"action": "MOVE_POSE", "frame_id": "unknown", "position": [0.1, 0.2, 0.3]},
            ]
        })

        result = RobotRuntime().execute_semantic_program(_FakeRobot(), program, execute=False)

        self.assertFalse(result.success)
        self.assertIn("TASK_FRAME_TF_UNAVAILABLE", result.failure_reason)


if __name__ == "__main__":
    unittest.main()
