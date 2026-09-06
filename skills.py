from dataclasses import dataclass
from typing import List, Any
import numpy as np


@dataclass
class MoveTCP:
    position: np.ndarray
    frame: str = "robot_base"
    tolerance: float = 0.01
    timeout: float = 5.0
    duration: float = 3.0


@dataclass
class MoveJoints:
    positions: np.ndarray
    duration: float = 3.0


@dataclass
class Grasp:
    width: float = 0.0
    tolerance: float = 0.004
    timeout: float = 3.0


@dataclass
class Release:
    width: float = 0.08
    tolerance: float = 0.004
    timeout: float = 3.0


@dataclass
class TaskSequence:
    skills: List[Any]
    stop_on_failure: bool = True


@dataclass
class TaskResult:
    success: bool
    results: List[Any]
    failed_step: int | None = None
    failure_reason: str | None = None

@dataclass
class GraspObject:
    object_id: str
    strategy: str = "top"
    approach_height: float = 0.08
    lift_height: float = 0.08
