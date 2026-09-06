from dataclasses import dataclass
from typing import Any, Dict, List

from shared_detector import (
    SharedOpenVocabularyDetector,
)
from perception_snapshot import (
    MultiViewSnapshot,
)


@dataclass
class CameraInferenceResult:
    camera: str
    timestamp: float
    detections: List[Any]


class InferenceManager:
    """
    Perception inference over immutable sensor snapshots.

    Important:
    - Does not subscribe to ROS topics.
    - Does not spin ROS nodes.
    - Does not command robot motion.
    - Loads one shared detector.
    - Processes frozen camera frames only.
    """

    def __init__(
        self,
        detector=None,
        box_threshold=0.20,
        text_threshold=0.20,
    ):
        self.detector = (
            detector
            if detector is not None
            else SharedOpenVocabularyDetector(
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
        )

    def detect_camera(
        self,
        snapshot,
        query,
    ):
        detections = self.detector.detect(
            snapshot.rgb,
            query,
        )

        return CameraInferenceResult(
            camera=snapshot.camera,
            timestamp=(
                snapshot.observation_timestamp
            ),
            detections=detections,
        )

    def detect_multiview(
        self,
        snapshot: MultiViewSnapshot,
        query: str,
    ) -> Dict[str, CameraInferenceResult]:

        if not snapshot.valid:
            raise RuntimeError(
                f"INVALID_MULTIVIEW_SNAPSHOT: "
                f"{snapshot.reason}"
            )

        results = {}

        # GPU inference remains serialized intentionally.
        # Camera acquisition is concurrent; inference ownership
        # stays deterministic.
        for camera in (
            "main",
            "side",
            "wrist",
        ):
            camera_snapshot = snapshot.get(
                camera
            )

            if camera_snapshot is None:
                continue

            results[camera] = (
                self.detect_camera(
                    camera_snapshot,
                    query,
                )
            )

        return results
