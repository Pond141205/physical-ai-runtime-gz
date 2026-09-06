from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class RobotState:
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    tcp_position: np.ndarray


@dataclass
class SkillResult:
    skill: str
    success: bool
    target: np.ndarray
    actual: np.ndarray
    error: float
    duration: float
    failure_reason: Optional[str] = None