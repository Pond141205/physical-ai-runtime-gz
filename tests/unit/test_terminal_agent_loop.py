import unittest
from types import SimpleNamespace

import numpy as np

from physical_ai_runtime.perception.scene_state import SceneObject, SceneState
from scripts.physical_ai_terminal import resolve_agent_turn


class _Conversation:
    def __init__(self):
        self.contexts = []
        self.feedback = []

    def respond(self, text, runtime_context=None):
        self.contexts.append(runtime_context)
        if runtime_context is None:
            return SimpleNamespace(
                mode="observe",
                query="cube",
                message="Observing cube",
                program=None,
            )
        return SimpleNamespace(
            mode="chat",
            query=None,
            message="Geometry received",
            program=None,
        )

    def record_runtime_feedback(self, feedback):
        self.feedback.append(feedback)


class _Robot:
    base_frame = "portable_base"

    def get_tcp_pose(self):
        return (
            np.array([0.3, 0.1, 0.5]),
            np.array([0.0, 0.0, 0.0, 1.0]),
        )


class _Perception:
    def __init__(self):
        self.queries = []

    def observe_manipulation_target(self, *, query, timeout):
        self.queries.append((query, timeout))
        return {
            "success": True,
            "camera": "main",
            "reason": "MANIPULATION_GEOMETRY_AVAILABLE",
            "scene": SceneState(objects={
                query: SceneObject(
                    object_id=query,
                    position_world=np.array([0.8, -0.3, 0.75]),
                    position_robot=np.array([0.2, -0.3, 0.75]),
                    depth=0.7,
                    size_xyz=np.array([0.04, 0.04, 0.04]),
                    height=0.04,
                    support_z=0.73,
                ),
            }),
        }


class TerminalAgentLoopTest(unittest.TestCase):
    def test_observation_is_supplied_as_verified_metric_context(self):
        conversation = _Conversation()
        perception = _Perception()

        turn = resolve_agent_turn(
            conversation,
            "pick up the cube",
            robot=_Robot(),
            perception_manager=perception,
        )

        self.assertEqual(turn.mode, "chat")
        self.assertEqual(perception.queries, [("cube", 5.0)])
        context = conversation.contexts[-1]
        self.assertTrue(context["geometry_verified"])
        self.assertFalse(context["motion_authorized"])
        self.assertEqual(context["robot_base_frame"], "portable_base")
        self.assertEqual(
            context["object"]["position_robot"],
            [0.2, -0.3, 0.75],
        )
        self.assertEqual(
            conversation.feedback[-1]["event"],
            "OBJECT_OBSERVATION",
        )

    def test_repeated_observe_request_fails_closed(self):
        class _RepeatingConversation(_Conversation):
            def respond(self, text, runtime_context=None):
                return SimpleNamespace(
                    mode="observe",
                    query="cube",
                    message="Again",
                    program=None,
                )

        with self.assertRaisesRegex(
            RuntimeError,
            "CONVERSATION_OBSERVATION_LOOP_LIMIT",
        ):
            resolve_agent_turn(
                _RepeatingConversation(),
                "pick up the cube",
                robot=_Robot(),
                perception_manager=_Perception(),
            )


if __name__ == "__main__":
    unittest.main()
