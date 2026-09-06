from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from physical_ai_runtime.perception.spatial_consistency import SpatialConsensus


VISIBLE_STATUSES = {
    "OBJECT_VISIBLE",
}

OCCLUDED_STATUSES = {
    "SELF_OCCLUDED",
    "OBJECT_OCCLUDED",
}

NOT_VISIBLE_STATUSES = {
    "OBJECT_NOT_DETECTED",
}


@dataclass
class FusedCameraEvidence:
    camera: str

    ai_target_visible: bool
    ai_robot_occlusion: bool
    ai_usable: bool
    ai_confidence: float

    geometry_status: str
    geometry_valid: bool

    spatial_consistent: bool
    spatial_reason: str
    spatial_support_count: int

    accepted: bool
    score: float
    reason: str

    def to_dict(self):
        return asdict(self)


class EvidenceFusion:
    """
    Physical-AI perception fusion.

    Authority order:

      deterministic geometry
          ↓
      spatial / temporal consistency
          ↓
      multimodal AI evidence
          ↓
      ranking only

    AI cannot override failed geometry or failed
    spatial consensus.
    """

    CAMERA_NAMES = (
        "main",
        "side",
        "wrist",
    )

    def __init__(
        self,
        min_ai_confidence: float = 0.60,
        single_view_ai_confidence: float = 0.85,
        spatial_consensus: Optional[
            SpatialConsensus
        ] = None,
    ):
        self.min_ai_confidence = float(
            min_ai_confidence
        )

        self.single_view_ai_confidence = float(
            single_view_ai_confidence
        )

        self.spatial_consensus = (
            spatial_consensus
            or SpatialConsensus()
        )

    @staticmethod
    def _clip(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0

        return max(
            0.0,
            min(1.0, value),
        )

    def fuse(
        self,
        vision: Dict[str, Any],
        geometry: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:

        spatial = (
            self.spatial_consensus.evaluate(
                geometry
            )
        )

        spatial_results = spatial.get(
            "camera_results",
            {},
        )

        vision_cameras = vision.get(
            "cameras",
            {},
        )

        mode = spatial.get(
            "mode",
            "NONE",
        )

        results = {}

        for camera in self.CAMERA_NAMES:
            ai = vision_cameras.get(
                camera,
                {},
            )

            geo = geometry.get(
                camera,
                {},
            )

            spatial_item = (
                spatial_results.get(
                    camera,
                    {},
                )
            )

            ai_visible = bool(
                ai.get(
                    "target_visible",
                    False,
                )
            )

            ai_occluded = bool(
                ai.get(
                    "robot_occlusion",
                    False,
                )
            )

            ai_usable = bool(
                ai.get(
                    "usable",
                    False,
                )
            )

            ai_confidence = self._clip(
                ai.get(
                    "confidence",
                    0.0,
                )
            )

            status = str(
                geo.get(
                    "status",
                    "UNKNOWN",
                )
            )

            geometry_valid = (
                status in VISIBLE_STATUSES
            )

            spatial_consistent = bool(
                spatial_item.get(
                    "consistent",
                    False,
                )
            )

            spatial_reason = str(
                spatial_item.get(
                    "reason",
                    "NO_SPATIAL_RESULT",
                )
            )

            support_count = int(
                spatial_item.get(
                    "support_count",
                    0,
                )
            )

            required_ai_confidence = (
                self.single_view_ai_confidence
                if mode == "SINGLE_VIEW"
                else self.min_ai_confidence
            )

            accepted = (
                geometry_valid
                and spatial_consistent
                and ai_visible
                and ai_usable
                and ai_confidence
                >= required_ai_confidence
            )

            if not geometry_valid:
                if status in OCCLUDED_STATUSES:
                    reason = (
                        "REJECT_GEOMETRY_OCCLUDED"
                    )

                elif status in NOT_VISIBLE_STATUSES:
                    reason = (
                        "REJECT_GEOMETRY_NOT_VISIBLE"
                    )

                else:
                    reason = (
                        "REJECT_GEOMETRY_UNVERIFIED"
                    )

            elif not spatial_consistent:
                reason = (
                    "REJECT_"
                    + spatial_reason
                )

            elif not ai_visible:
                reason = (
                    "REJECT_AI_NOT_VISIBLE"
                )

            elif not ai_usable:
                reason = (
                    "REJECT_AI_VIEW_UNUSABLE"
                )

            elif (
                ai_confidence
                < required_ai_confidence
            ):
                reason = (
                    "REJECT_AI_LOW_CONFIDENCE"
                )

            else:
                reason = "ACCEPTED"

            if accepted:
                # Geometry and spatial consistency already passed.
                # AI is now used only for ranking.
                score = (
                    0.70
                    + 0.30 * ai_confidence
                )

                # Slight preference for observations supported
                # by more independent cameras.
                score += min(
                    0.05,
                    0.01 * max(
                        0,
                        support_count - 1,
                    ),
                )

            else:
                score = 0.0

            results[
                camera
            ] = FusedCameraEvidence(
                camera=camera,

                ai_target_visible=ai_visible,
                ai_robot_occlusion=ai_occluded,
                ai_usable=ai_usable,
                ai_confidence=ai_confidence,

                geometry_status=status,
                geometry_valid=geometry_valid,

                spatial_consistent=(
                    spatial_consistent
                ),
                spatial_reason=(
                    spatial_reason
                ),
                spatial_support_count=(
                    support_count
                ),

                accepted=accepted,
                score=score,
                reason=reason,
            )

        candidates = [
            result
            for result in results.values()
            if result.accepted
        ]

        candidates.sort(
            key=lambda x: x.score,
            reverse=True,
        )

        best = (
            candidates[0]
            if candidates
            else None
        )

        return {
            "target_visible": (
                best is not None
            ),

            "best_camera": (
                best.camera
                if best
                else None
            ),

            "cameras": {
                name: result.to_dict()
                for name, result
                in results.items()
            },

            "spatial_consensus": spatial,

            "safe_visual_evidence": (
                best is not None
                and spatial.get(
                    "safe_spatial_consensus",
                    False,
                )
            ),

            "reason": (
                "FUSED_VISUAL_EVIDENCE_ACCEPTED"
                if best
                else
                "NO_CAMERA_PASSED_EVIDENCE_FUSION"
            ),
        }
