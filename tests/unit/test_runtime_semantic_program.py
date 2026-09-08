import unittest
from types import SimpleNamespace

import numpy as np

from physical_ai_runtime.ai.semantic_manipulation_task import (
    SemanticTaskProgram,
)
from physical_ai_runtime.core.feedback import SemanticFeedback
from physical_ai_runtime.core.runtime import RobotRuntime
from physical_ai_runtime.core.skills import GraspObject


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


class _FakePerception:
    def __init__(self):
        self.scene = object()

    def observe_manipulation_target(self, *, query, timeout):
        return {
            "success": True,
            "scene": self.scene,
            "camera": "main",
        }


class _RecordingRuntime(RobotRuntime):
    def execute_grasp_attempt(self, robot, skill, **kwargs):
        self.grasp_kwargs = kwargs
        return SemanticFeedback(
            state="OBJECT_LIFTED",
            object_id=skill.object_id,
        )


class _ControlRobot:
    def __init__(self):
        self.calls = []

    def stop(self):
        self.calls.append("stop")
        return {"success": True, "failure_reason": None}

    def get_capabilities(self):
        return SimpleNamespace(grasp=True)

    def move_gripper(self, target, *, tolerance, timeout):
        self.calls.append(("gripper", target, tolerance, timeout))
        return {"success": True, "failure_reason": None}


class _WorldLiftPlanningRobot:
    base_frame = "rotated_robot_base"

    def __init__(self):
        self.planned_target = None
        self.planned_orientation = None

    def get_tcp_pose(self):
        return (
            np.array([0.4, -0.2, 0.7]),
            np.array([0.0, 0.0, 0.0, 1.0]),
        )

    def get_world_up_vector_in_base(self):
        # Deliberately not base +Z: this represents a rotated embodiment.
        return np.array([0.6, 0.0, 0.8])

    def plan_tcp_pose(self, *, target, orientation):
        self.planned_target = np.asarray(target, dtype=float)
        self.planned_orientation = np.asarray(orientation, dtype=float)
        return {
            "success": True,
            "trajectory": object(),
            "planned_at_monotonic": 1.0,
        }


class _LiftExecutionRuntime(RobotRuntime):
    def __init__(self):
        super().__init__()
        self.authorized_trajectory = None
        self.perception_manager = None

    def _execute_authorized_trajectory(
        self,
        robot,
        trajectory,
        planned_at_monotonic,
    ):
        self.authorized_trajectory = (
            trajectory,
            planned_at_monotonic,
        )
        return SimpleNamespace(
            success=True,
            error=0.0,
            failure_reason=None,
        ), None

    def verify_lifted_object(
        self,
        robot,
        skill,
        timeout,
        perception_manager,
    ):
        self.perception_manager = perception_manager
        return {
            "success": True,
            "observed_world": [0.0, 0.0, 0.8],
            "vertical_displacement": 0.08,
            "minimum_evidence": 0.02,
        }


class RuntimeSemanticProgramTest(unittest.TestCase):
    def test_plans_all_gesture_steps_in_adapter_base_frame(self):
        program = SemanticTaskProgram.from_model_output({
            "steps": [
                {"action": "MOVE_TO", "frame_id": "test_base", "position": [0.1, 0.2, 0.3]},
                {"action": "MOVE_TO", "frame_id": "test_base", "position": [0.1, -0.2, 0.3]},
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
                {"action": "MOVE_TO", "frame_id": "unknown", "position": [0.1, 0.2, 0.3]},
            ]
        })

        result = RobotRuntime().execute_semantic_program(_FakeRobot(), program, execute=False)

        self.assertFalse(result.success)
        self.assertIn("TASK_FRAME_TF_UNAVAILABLE", result.failure_reason)

    def test_public_program_rejects_pick(self):
        with self.assertRaisesRegex(
            ValueError,
            "SEMANTIC_TASK_UNSUPPORTED_ACTION",
        ):
            SemanticTaskProgram.from_model_output({
                "steps": [{"action": "PICK", "object_id": "cube"}],
            })

    def test_stop_and_open_use_common_runtime_controls(self):
        program = SemanticTaskProgram.from_model_output({
            "steps": [
                {"action": "STOP"},
                {"action": "OPEN"},
            ],
        })
        robot = _ControlRobot()

        result = RobotRuntime().execute_semantic_program(
            robot,
            program,
            execute=True,
        )

        self.assertTrue(result.success)
        self.assertEqual(robot.calls[0], "stop")
        self.assertEqual(robot.calls[1][0], "gripper")

    def test_plan_only_stop_and_open_do_not_call_adapter(self):
        program = SemanticTaskProgram.from_model_output({
            "steps": [
                {"action": "STOP"},
                {"action": "OPEN"},
            ],
        })
        robot = _ControlRobot()

        result = RobotRuntime().execute_semantic_program(
            robot,
            program,
            execute=False,
        )

        self.assertTrue(result.success)
        self.assertEqual(robot.calls, [])
        self.assertTrue(result.results[0]["planned"])


class RuntimeWorldLiftPlanningTest(unittest.TestCase):
    def test_lift_uses_measured_tcp_and_adapter_world_up(self):
        robot = _WorldLiftPlanningRobot()

        result = RobotRuntime()._plan_world_vertical_motion(
            robot,
            distance=0.08,
        )

        self.assertTrue(result["success"])
        np.testing.assert_allclose(
            robot.planned_target,
            [0.448, -0.2, 0.764],
        )
        np.testing.assert_allclose(
            result["start_tcp"],
            [0.4, -0.2, 0.7],
        )
        np.testing.assert_allclose(
            result["world_up_in_base"],
            [0.6, 0.0, 0.8],
        )

    def test_lift_fails_closed_without_embodiment_frame_support(self):
        result = RobotRuntime()._plan_world_vertical_motion(
            _FakeRobot(),
            distance=0.08,
        )

        self.assertFalse(result["success"])
        self.assertEqual(
            result["failure_reason"],
            "WORLD_UP_VECTOR_UNAVAILABLE",
        )

    def test_execute_lift_uses_authorized_world_lift_plan(self):
        robot = _WorldLiftPlanningRobot()
        robot.current_gripper_position = np.array([0.02, 0.02])
        runtime = _LiftExecutionRuntime()
        runtime.active_grasp_context = {"object_id": "cube"}
        perception_manager = object()

        feedback = runtime._execute_verified_grasp_vertical_motion(
            robot,
            GraspObject("cube", lift_height=0.08),
            perception_manager=perception_manager,
        )

        self.assertEqual(feedback.state, "OBJECT_LIFTED")
        self.assertIsNotNone(runtime.authorized_trajectory)
        self.assertIs(
            runtime.perception_manager,
            perception_manager,
        )
        np.testing.assert_allclose(
            robot.planned_target,
            [0.448, -0.2, 0.764],
        )


if __name__ == "__main__":
    unittest.main()
