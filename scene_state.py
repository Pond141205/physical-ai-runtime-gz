from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class SceneObject:
    object_id: str
    position_world: np.ndarray
    position_robot: np.ndarray
    depth: float

    # Optional geometry estimated by perception
    size_xyz: Optional[np.ndarray] = None
    height: Optional[float] = None
    support_z: Optional[float] = None


@dataclass
class SceneState:
    objects: Dict[str, SceneObject]
