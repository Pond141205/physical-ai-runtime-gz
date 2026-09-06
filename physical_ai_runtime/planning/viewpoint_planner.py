from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class ViewpointCandidate:
    candidate_id: str

    camera_position_world: np.ndarray
    target_position_world: np.ndarray

    distance_to_target: float

    semantic_direction: str
    score: float

    valid: bool
    reason: str


@dataclass(frozen=True)
class ViewpointPlan:
    candidates: List[ViewpointCandidate]

    selected_candidate: Optional[ViewpointCandidate]

    safe_to_plan_motion: bool

    reason: str


class ViewpointPlanner:
    """
    Generates geometric camera-view candidates around a target.

    IMPORTANT:
      - No robot motion
      - No MoveIt execution
      - No motion authorization

    This layer only proposes candidate camera positions.

    Later:
        candidate
            ↓
        wrist-camera -> TCP transform
            ↓
        MoveIt plan-only
            ↓
        runtime safety validation
            ↓
        optional execution
    """

    def __init__(
        self,
        *,
        preferred_distance_m: float = 0.45,
        min_distance_m: float = 0.25,
        max_distance_m: float = 0.70,
        minimum_camera_z: float = 0.80,
        maximum_camera_z: float = 1.40,
    ):
        self.preferred_distance_m = float(
            preferred_distance_m
        )

        self.min_distance_m = float(
            min_distance_m
        )

        self.max_distance_m = float(
            max_distance_m
        )

        self.minimum_camera_z = float(
            minimum_camera_z
        )

        self.maximum_camera_z = float(
            maximum_camera_z
        )

    def _candidate(
        self,
        *,
        candidate_id,
        target,
        offset,
        semantic_direction,
        base_score,
    ):
        position = (
            np.asarray(
                target,
                dtype=float,
            )
            + np.asarray(
                offset,
                dtype=float,
            )
        )

        delta = position - target

        distance = float(
            np.linalg.norm(delta)
        )

        valid = True
        reason = "GEOMETRICALLY_VALID"

        if not np.all(
            np.isfinite(position)
        ):
            valid = False
            reason = "NON_FINITE_POSITION"

        elif (
            distance
            < self.min_distance_m
        ):
            valid = False
            reason = "TOO_CLOSE_TO_TARGET"

        elif (
            distance
            > self.max_distance_m
        ):
            valid = False
            reason = "TOO_FAR_FROM_TARGET"

        elif (
            position[2]
            < self.minimum_camera_z
        ):
            valid = False
            reason = "CAMERA_TOO_LOW"

        elif (
            position[2]
            > self.maximum_camera_z
        ):
            valid = False
            reason = "CAMERA_TOO_HIGH"

        distance_error = abs(
            distance
            - self.preferred_distance_m
        )

        score = (
            float(base_score)
            - distance_error
        )

        if not valid:
            score = 0.0

        return ViewpointCandidate(
            candidate_id=candidate_id,
            camera_position_world=position,
            target_position_world=np.asarray(
                target,
                dtype=float,
            ),
            distance_to_target=distance,
            semantic_direction=semantic_direction,
            score=score,
            valid=valid,
            reason=reason,
        )

    def generate_candidates(
        self,
        scene_state,
        *,
        object_id: str,
        semantic_intent: Optional[str] = None,
    ) -> ViewpointPlan:

        if scene_state is None:
            return ViewpointPlan(
                candidates=[],
                selected_candidate=None,
                safe_to_plan_motion=False,
                reason="SCENE_STATE_MISSING",
            )

        objects = getattr(
            scene_state,
            "objects",
            None,
        )

        if not objects:
            return ViewpointPlan(
                candidates=[],
                selected_candidate=None,
                safe_to_plan_motion=False,
                reason="SCENE_EMPTY",
            )

        obj = objects.get(
            object_id
        )

        if obj is None:
            return ViewpointPlan(
                candidates=[],
                selected_candidate=None,
                safe_to_plan_motion=False,
                reason="TARGET_OBJECT_MISSING",
            )

        target = np.asarray(
            obj.position_world,
            dtype=float,
        )

        if (
            target.shape != (3,)
            or not np.all(
                np.isfinite(target)
            )
        ):
            return ViewpointPlan(
                candidates=[],
                selected_candidate=None,
                safe_to_plan_motion=False,
                reason="TARGET_POSITION_INVALID",
            )

        #
        # These are semantic observation positions,
        # NOT robot TCP commands.
        #
        # We bias above the target because the table
        # occupies the lower half-space.
        #
        candidate_specs = [
            (
                "front_high",
                np.array(
                    [-0.32, 0.00, 0.30]
                ),
                "front-above",
                1.00,
            ),
            (
                "left_high",
                np.array(
                    [0.00, 0.32, 0.30]
                ),
                "left-above",
                0.95,
            ),
            (
                "right_high",
                np.array(
                    [0.00, -0.32, 0.30]
                ),
                "right-above",
                0.95,
            ),
            (
                "rear_high",
                np.array(
                    [0.32, 0.00, 0.30]
                ),
                "rear-above",
                0.85,
            ),
            (
                "top",
                np.array(
                    [0.00, 0.00, 0.45]
                ),
                "top-down",
                0.80,
            ),
        ]

        candidates = []

        for (
            candidate_id,
            offset,
            semantic_direction,
            base_score,
        ) in candidate_specs:

            candidate = self._candidate(
                candidate_id=candidate_id,
                target=target,
                offset=offset,
                semantic_direction=(
                    semantic_direction
                ),
                base_score=base_score,
            )

            candidates.append(
                candidate
            )

        valid_candidates = [
            candidate
            for candidate in candidates
            if candidate.valid
        ]

        if not valid_candidates:
            return ViewpointPlan(
                candidates=candidates,
                selected_candidate=None,
                safe_to_plan_motion=False,
                reason=(
                    "NO_GEOMETRICALLY_VALID_CANDIDATE"
                ),
            )

        selected = max(
            valid_candidates,
            key=lambda c: c.score,
        )

        return ViewpointPlan(
            candidates=candidates,
            selected_candidate=selected,

            #
            # IMPORTANT:
            #
            # True here means:
            # "safe to continue into PLAN-ONLY validation"
            #
            # It does NOT mean safe to execute motion.
            #
            safe_to_plan_motion=True,

            reason=(
                "GEOMETRIC_CANDIDATE_AVAILABLE"
            ),
        )
