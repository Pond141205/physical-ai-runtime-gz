"""Live sensor regression for the conversational observe handshake.

Requires the canonical full stack. This test performs no robot motion.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import rclpy

from physical_ai_runtime.adapters.gazebo_panda_adapter import (
    GazeboPandaAdapter,
)
from physical_ai_runtime.perception.camera_manager import CameraManager
from scripts.physical_ai_terminal import resolve_agent_turn


class _ObserveThenStopConversation:
    def __init__(self):
        self.context = None
        self.feedback = []

    def respond(self, text, runtime_context=None):
        if runtime_context is None:
            return SimpleNamespace(
                mode="observe",
                query="cube",
                message="Requesting verified geometry",
                program=None,
            )
        self.context = runtime_context
        return SimpleNamespace(
            mode="chat",
            query=None,
            message="Geometry received; no motion requested",
            program=None,
        )

    def record_runtime_feedback(self, feedback):
        self.feedback.append(feedback)


class TerminalObservationLoopIntegrationTest(unittest.TestCase):
    def test_live_cube_geometry_reaches_agent_without_motion_authority(self):
        robot = None
        manager = None
        rclpy.init()

        try:
            robot = GazeboPandaAdapter()
            manager = CameraManager(query="cube")
            manager.start_sensor_streams()

            preflight = robot.preflight_check(timeout=10.0)
            self.assertTrue(preflight["success"], str(preflight))

            conversation = _ObserveThenStopConversation()
            turn = resolve_agent_turn(
                conversation,
                "pick up the cube",
                robot=robot,
                perception_manager=manager,
            )

            self.assertEqual(turn.mode, "chat")
            context = conversation.context
            self.assertIsNotNone(context)
            self.assertTrue(context["geometry_verified"], str(context))
            self.assertFalse(context["motion_authorized"])
            self.assertEqual(
                context["robot_base_frame"],
                robot.base_frame,
            )

            obj = context["object"]
            self.assertEqual(obj["object_id"], "cube")
            self.assertTrue(
                np.all(np.isfinite(obj["position_robot"]))
            )
            self.assertTrue(
                np.all(np.isfinite(obj["position_world"]))
            )

            print("TERMINAL_OBSERVE_INTEGRATION")
            print("camera=" + str(context.get("camera")))
            print(
                "position_robot="
                + str(obj["position_robot"])
            )
            print(
                "position_world="
                + str(obj["position_world"])
            )
            print(
                "geometry_verified="
                + str(context["geometry_verified"])
            )
            print(
                "motion_authorized="
                + str(context["motion_authorized"])
            )
        finally:
            if manager is not None:
                manager.close()
            if robot is not None:
                robot.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
