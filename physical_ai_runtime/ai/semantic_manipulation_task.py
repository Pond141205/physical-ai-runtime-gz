"""Validated AI task-space programs at the AI-to-runtime boundary."""

from dataclasses import dataclass
from typing import Optional


PICK = "PICK"
PICK_AND_PLACE = "PICK_AND_PLACE"
HOLD = "HOLD"
MOVE_POSE = "MOVE_POSE"
GRASP = "GRASP"
RELEASE = "RELEASE"
RELATIONS = {"beside", "on_top"}
TASK_ACTIONS = {PICK, PICK_AND_PLACE, HOLD, MOVE_POSE, GRASP, RELEASE}


def _vector(value, size, error):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(error)
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc


@dataclass(frozen=True)
class SemanticManipulationTask:
    action: str
    object_id: Optional[str] = None
    relation: Optional[str] = None
    reference_object_id: Optional[str] = None
    frame_id: Optional[str] = None
    position: Optional[tuple[float, float, float]] = None
    orientation: Optional[tuple[float, float, float, float]] = None

    @classmethod
    def from_model_output(cls, value):
        if not isinstance(value, dict):
            raise ValueError("SEMANTIC_TASK_INVALID_SCHEMA")
        position = value.get("position")
        orientation = value.get("orientation")
        task = cls(
            action=str(value.get("action", "")).upper(),
            object_id=None if value.get("object_id") is None else str(value["object_id"]).strip().lower(),
            relation=None if value.get("relation") is None else str(value["relation"]).strip().lower(),
            reference_object_id=None if value.get("reference_object_id") is None else str(value["reference_object_id"]).strip().lower(),
            frame_id=None if value.get("frame_id") is None else str(value["frame_id"]).strip(),
            position=None if position is None else _vector(position, 3, "SEMANTIC_TASK_INVALID_POSITION"),
            orientation=None if orientation is None else _vector(orientation, 4, "SEMANTIC_TASK_INVALID_ORIENTATION"),
        )
        task.validate()
        return task

    def validate(self):
        if self.action not in TASK_ACTIONS:
            raise ValueError("SEMANTIC_TASK_UNSUPPORTED_ACTION")
        if self.action == MOVE_POSE:
            if not self.frame_id or self.position is None:
                raise ValueError("SEMANTIC_TASK_POSE_REQUIRED")
            if self.object_id or self.relation or self.reference_object_id:
                raise ValueError("SEMANTIC_TASK_UNEXPECTED_OBJECT_FIELDS")
            return
        if self.action in {GRASP, RELEASE}:
            if self.action == GRASP and not self.object_id:
                raise ValueError("SEMANTIC_TASK_INVALID_OBJECT")
            if self.relation or self.reference_object_id or self.position or self.orientation:
                raise ValueError("SEMANTIC_TASK_UNEXPECTED_RELATION")
            return
        if not self.object_id or any(char.isspace() for char in self.object_id):
            raise ValueError("SEMANTIC_TASK_INVALID_OBJECT")
        if self.action == PICK_AND_PLACE:
            if self.relation not in RELATIONS:
                raise ValueError("SEMANTIC_TASK_INVALID_RELATION")
            if not self.reference_object_id or self.reference_object_id == self.object_id:
                raise ValueError("SEMANTIC_TASK_INVALID_REFERENCE")
        elif self.relation is not None or self.reference_object_id is not None:
            raise ValueError("SEMANTIC_TASK_UNEXPECTED_RELATION")


@dataclass(frozen=True)
class SemanticTaskProgram:
    steps: tuple[SemanticManipulationTask, ...]

    @classmethod
    def from_model_output(cls, value):
        if not isinstance(value, dict) or not isinstance(value.get("steps"), list):
            raise ValueError("SEMANTIC_PROGRAM_INVALID_SCHEMA")
        if not value["steps"]:
            raise ValueError("SEMANTIC_PROGRAM_EMPTY")
        return cls(tuple(SemanticManipulationTask.from_model_output(step) for step in value["steps"]))
