from threading import Lock

from physical_ai_runtime.perception.object_detection.open_vocab_detector import (
    OpenVocabularyDetector,
)


class SharedOpenVocabularyDetector:
    """
    One detector instance shared by all camera observers.

    Sensor acquisition may happen concurrently, but GPU inference
    is serialized here until a dedicated batched detector service
    is introduced.

    This avoids:
      - loading GroundingDINO once per camera
      - multiple copies of model weights in VRAM
      - unsafe simultaneous access to one model instance
    """

    def __init__(
        self,
        box_threshold=0.20,
        text_threshold=0.20,
    ):
        self._detector = OpenVocabularyDetector(
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        self._lock = Lock()

    def detect(self, image, query):
        with self._lock:
            return self._detector.detect(
                image,
                query,
            )
