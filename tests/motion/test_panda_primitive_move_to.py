import unittest

import numpy as np
import rclpy

from physical_ai_runtime.adapters.gazebo_panda_adapter import (
    GazeboPandaAdapter,
)
from physical_ai_runtime.ai.semantic_manipulation_task import (
    SemanticTaskProgram,
)
from physical_ai_runtime.core.runtime import RobotRuntime


class PandaPrimitiveMoveToTest(unittest.TestCase):
    def test_move_to_uses_public_runtime_boundary(self):
        robot = None
        rclpy.init()

        try:
            robot = GazeboPandaAdapter()
            preflight = robot.preflight_check(timeout=10.0)
            self.assertTrue(preflight["success"], str(preflight))

            target = np.array([0.45, 0.15, 0.45], dtype=float)
            program = SemanticTaskProgram.from_model_output({
                "steps": [{
                    "action": "MOVE_TO",
                    "frame_id": robot.base_frame,
                    "position": target.tolist(),
                }],
            })
            result = RobotRuntime().execute_semantic_program(
                robot,
                program,
                execute=True,
            )
            actual, _ = robot.get_tcp_pose(timeout=5.0)
            error = float(np.linalg.norm(actual - target))
            timeout = bool(
                result.failure_reason
                and "TIMEOUT" in result.failure_reason
            )

            print("PANDA_PRIMITIVE_MOVE_TO")
            print("target_tcp=" + str(target.tolist()))
            print("actual_tcp=" + str(actual.tolist()))
            print("euclidean_error_m=" + str(error))
            print("timeout=" + str(timeout))
            print("result_success=" + str(result.success))
            print("failure_reason=" + str(result.failure_reason))

            self.assertTrue(result.success, str(result.failure_reason))
            self.assertFalse(timeout)
            self.assertLessEqual(error, 0.02)
        finally:
            if robot is not None:
                robot.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
