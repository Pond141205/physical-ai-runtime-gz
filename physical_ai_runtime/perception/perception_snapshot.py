from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class CameraSnapshot:
    camera: str

    rgb: np.ndarray
    depth: np.ndarray

    rgb_timestamp: float
    depth_timestamp: float
    observation_timestamp: float

    fx: float
    fy: float
    cx: float
    cy: float

    camera_frame: str
    sensor_frame: Optional[str] = None


@dataclass(frozen=True)
class MultiViewSnapshot:
    cameras: Dict[str, CameraSnapshot]

    cross_camera_skew_s: float
    valid: bool
    reason: str

    def get(self, camera: str):
        return self.cameras.get(camera)
