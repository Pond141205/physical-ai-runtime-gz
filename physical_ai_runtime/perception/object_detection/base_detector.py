from abc import ABC, abstractmethod
from typing import List
import numpy as np

from .detection_types import Detection


class BaseObjectDetector(ABC):

    @abstractmethod
    def detect(
        self,
        image_rgb: np.ndarray,
        query: str,
    ) -> List[Detection]:
        raise NotImplementedError
