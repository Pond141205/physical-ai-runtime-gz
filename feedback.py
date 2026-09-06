from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SemanticFeedback:
    state: str

    object_id: Optional[str] = None

    message: Optional[str] = None

    confidence: Optional[float] = None

    tcp_pose: Optional[np.ndarray] = None

    object_pose: Optional[np.ndarray] = None

    gripper_positions: Optional[np.ndarray] = None

    metrics: Optional[Dict[str, Any]] = None

    available_actions: Optional[List[str]] = None


GRASP_READY = "GRASP_READY"

OBJECT_VISIBLE = "OBJECT_VISIBLE"
OBJECT_OCCLUDED = "OBJECT_OCCLUDED"
OBJECT_LOST = "OBJECT_LOST"

CONTACT_DETECTED = "CONTACT_DETECTED"
CONTACT_ASYMMETRIC = "CONTACT_ASYMMETRIC"
CONTACT_ACCEPTABLE = "CONTACT_ACCEPTABLE"
NO_CONTACT = "NO_CONTACT"

MOTION_REACHED = "MOTION_REACHED"
MOTION_FAILED = "MOTION_FAILED"

READY_REACHED = "READY_REACHED"
READY_FAILED = "READY_FAILED"

OBJECT_LIFTED = "OBJECT_LIFTED"
OBJECT_NOT_LIFTED = "OBJECT_NOT_LIFTED"

REPLAN_REQUIRED = "REPLAN_REQUIRED"
RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
