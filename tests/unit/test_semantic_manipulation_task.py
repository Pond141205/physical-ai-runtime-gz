import unittest

from physical_ai_runtime.ai.semantic_manipulation_task import (
    SemanticManipulationTask,
    SemanticTaskProgram,
)
from physical_ai_runtime.core.skills import PlaceObject


class SemanticManipulationTaskTest(unittest.TestCase):
    def test_pick_and_place_is_robot_independent(self):
        task = SemanticManipulationTask.from_model_output({
            "action": "pick_and_place",
            "object_id": "cube",
            "relation": "beside",
            "reference_object_id": "cylinder",
        })
        place = PlaceObject(task.object_id, task.relation, task.reference_object_id)
        self.assertEqual(place.reference_object_id, "cylinder")

    def test_accepts_arbitrary_task_space_pose(self):
        task = SemanticManipulationTask.from_model_output({
            "action": "move_pose",
            "frame_id": "world",
            "position": [0.31, -0.24, 0.62],
            "orientation": [0.0, 0.0, 0.0, 1.0],
        })
        self.assertEqual(task.frame_id, "world")
        self.assertEqual(task.position, (0.31, -0.24, 0.62))

    def test_accepts_gesture_as_multiple_runtime_validated_steps(self):
        program = SemanticTaskProgram.from_model_output({
            "steps": [
                {"action": "MOVE_POSE", "frame_id": "world", "position": [0.2, 0.1, 0.5]},
                {"action": "MOVE_POSE", "frame_id": "world", "position": [0.2, -0.1, 0.5]},
            ]
        })
        self.assertEqual(len(program.steps), 2)

    def test_rejects_direct_actuator_commands(self):
        with self.assertRaisesRegex(ValueError, "SEMANTIC_TASK_UNSUPPORTED_ACTION"):
            SemanticManipulationTask.from_model_output({
                "action": "SET_JOINT_TORQUE",
                "position": [1, 2, 3],
            })

    def test_place_requires_supported_relation_and_reference(self):
        with self.assertRaisesRegex(ValueError, "SEMANTIC_TASK_INVALID_REFERENCE"):
            SemanticManipulationTask.from_model_output({
                "action": "PICK_AND_PLACE",
                "object_id": "cube",
                "relation": "beside",
                "reference_object_id": "cube",
            })


if __name__ == "__main__":
    unittest.main()
