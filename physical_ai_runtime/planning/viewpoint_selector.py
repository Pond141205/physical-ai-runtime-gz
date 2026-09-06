from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ViewpointDecision:
    action: str
    selected_camera: Optional[str]
    reason: str

    target_visible: bool
    requires_new_viewpoint: bool

    preferred_semantic_intent: Optional[str]
    diagnostics: Dict


class ViewpointSelector:
    """
    Occlusion-aware viewpoint decision layer.

    This module DOES NOT command robot motion.

    It converts verified perception evidence into one of:

      USE_CURRENT_VIEW
      REQUEST_ALTERNATE_VIEW
      TARGET_OCCLUDED
      TARGET_NOT_FOUND
      PERCEPTION_UNVERIFIED

    A future ViewpointPlanner may consume semantic intent from
    this layer, but RobotRuntime remains responsible for safety
    and motion authorization.
    """

    def __init__(
        self,
        minimum_score=0.80,
    ):
        self.minimum_score = float(
            minimum_score
        )

    def select(
        self,
        perception_result,
    ):
        fusion = (
            perception_result.get(
                "fusion"
            )
            or {}
        )

        vision = (
            perception_result.get(
                "vision"
            )
            or {}
        )

        safe = bool(
            perception_result.get(
                "safe_visual_evidence",
                False,
            )
        )

        best_camera = (
            perception_result.get(
                "best_camera"
            )
        )

        fused_cameras = fusion.get(
            "cameras",
            {},
        )

        ai_cameras = vision.get(
            "cameras",
            {},
        )

        diagnostics = {
            "safe_visual_evidence": safe,
            "best_camera": best_camera,
            "camera_reasons": {},
        }

        for camera in (
            "main",
            "side",
            "wrist",
        ):
            fused = fused_cameras.get(
                camera,
                {},
            )

            ai = ai_cameras.get(
                camera,
                {},
            )

            diagnostics[
                "camera_reasons"
            ][camera] = {
                "accepted": fused.get(
                    "accepted",
                    False,
                ),
                "score": fused.get(
                    "score",
                    0.0,
                ),
                "fusion_reason": fused.get(
                    "reason",
                ),
                "geometry_status": fused.get(
                    "geometry_status",
                ),
                "ai_visible": ai.get(
                    "target_visible",
                    False,
                ),
                "ai_usable": ai.get(
                    "usable",
                    False,
                ),
                "ai_occlusion": ai.get(
                    "robot_occlusion",
                    False,
                ),
            }

        # --------------------------------------------------
        # Verified current view exists.
        # --------------------------------------------------

        if safe and best_camera:
            best = fused_cameras.get(
                best_camera,
                {},
            )

            score = float(
                best.get(
                    "score",
                    0.0,
                )
            )

            if score >= self.minimum_score:
                return ViewpointDecision(
                    action="USE_CURRENT_VIEW",
                    selected_camera=best_camera,
                    reason=(
                        "VERIFIED_CURRENT_VIEW_AVAILABLE"
                    ),
                    target_visible=True,
                    requires_new_viewpoint=False,
                    preferred_semantic_intent=None,
                    diagnostics=diagnostics,
                )

        # --------------------------------------------------
        # Determine whether evidence says target is occluded.
        # --------------------------------------------------

        occluded_cameras = []

        for camera in (
            "main",
            "side",
            "wrist",
        ):
            fused = fused_cameras.get(
                camera,
                {},
            )

            ai = ai_cameras.get(
                camera,
                {},
            )

            geometry_status = fused.get(
                "geometry_status"
            )

            if (
                geometry_status in (
                    "SELF_OCCLUDED",
                    "OBJECT_OCCLUDED",
                )
                or ai.get(
                    "robot_occlusion",
                    False,
                )
            ):
                occluded_cameras.append(
                    camera
                )

        if occluded_cameras:
            spatial_intent = vision.get(
                "spatial_intent"
            )

            return ViewpointDecision(
                action="REQUEST_ALTERNATE_VIEW",
                selected_camera=None,
                reason="TARGET_OCCLUDED_OR_VIEW_BLOCKED",
                target_visible=False,
                requires_new_viewpoint=True,
                preferred_semantic_intent=(
                    spatial_intent
                    or (
                        "obtain an unobstructed view "
                        "of the target"
                    )
                ),
                diagnostics={
                    **diagnostics,
                    "occluded_cameras":
                        occluded_cameras,
                },
            )

        # --------------------------------------------------
        # AI may report target visible but geometry cannot verify.
        # Never authorize from AI alone.
        # --------------------------------------------------

        ai_visible_anywhere = any(
            bool(
                ai_cameras.get(
                    camera,
                    {},
                ).get(
                    "target_visible",
                    False,
                )
            )
            for camera in (
                "main",
                "side",
                "wrist",
            )
        )

        if ai_visible_anywhere:
            return ViewpointDecision(
                action="PERCEPTION_UNVERIFIED",
                selected_camera=None,
                reason=(
                    "AI_VISIBILITY_WITHOUT_VERIFIED_GEOMETRY"
                ),
                target_visible=False,
                requires_new_viewpoint=True,
                preferred_semantic_intent=(
                    vision.get(
                        "spatial_intent"
                    )
                    or (
                        "obtain a geometrically "
                        "verifiable target view"
                    )
                ),
                diagnostics=diagnostics,
            )

        # --------------------------------------------------
        # Nothing currently sees target.
        # --------------------------------------------------

        return ViewpointDecision(
            action="TARGET_NOT_FOUND",
            selected_camera=None,
            reason="NO_CAMERA_VERIFIED_TARGET",
            target_visible=False,
            requires_new_viewpoint=True,
            preferred_semantic_intent=(
                vision.get(
                    "spatial_intent"
                )
                or (
                    "search for a viewpoint likely "
                    "to reveal the target"
                )
            ),
            diagnostics=diagnostics,
        )
