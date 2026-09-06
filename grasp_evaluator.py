from dataclasses import dataclass
from typing import Optional

import numpy as np

from feedback import (
    SemanticFeedback,
    CONTACT_DETECTED,
    CONTACT_ASYMMETRIC,
    CONTACT_ACCEPTABLE,
    NO_CONTACT,
)


@dataclass
class GraspEvaluationConfig:
    min_contact_position: float = 0.002
    acceptable_asymmetry: float = 0.006


class GraspEvaluator:
    """
    Converts low-level gripper feedback into semantic grasp state.

    This class:
    - does NOT command the robot
    - does NOT plan motion
    - does NOT assume a specific scene
    - only interprets feedback
    """

    def __init__(
        self,
        config: Optional[GraspEvaluationConfig] = None,
    ):
        self.config = (
            config
            if config is not None
            else GraspEvaluationConfig()
        )

    def evaluate_contact(
        self,
        object_id,
        gripper_positions,
    ):
        q = np.asarray(
            gripper_positions,
            dtype=float,
        )

        if q.size != 2:
            return SemanticFeedback(
                state=NO_CONTACT,
                object_id=object_id,
                message="Invalid gripper feedback.",
                gripper_positions=q,
                available_actions=[
                    "REOBSERVE",
                    "ABORT",
                ],
            )

        q1 = float(q[0])
        q2 = float(q[1])

        min_contact = (
            self.config.min_contact_position
        )

        asymmetry = abs(q1 - q2)

        estimated_offset = (
            q1 - q2
        ) / 2.0

        metrics = {
            "finger_1": q1,
            "finger_2": q2,
            "asymmetry": asymmetry,
            "estimated_tool_y_offset": estimated_offset,
        }

        #
        # Both fingers effectively closed:
        # likely no object was retained.
        #
        if (
            q1 <= min_contact
            and q2 <= min_contact
        ):
            return SemanticFeedback(
                state=NO_CONTACT,
                object_id=object_id,
                message=(
                    "Gripper closed without "
                    "persistent contact."
                ),
                gripper_positions=q,
                metrics=metrics,
                available_actions=[
                    "REOBSERVE",
                    "RETRY_GRASP",
                    "CHANGE_GRASP_STRATEGY",
                    "RETURN_READY",
                    "ABORT",
                ],
            )

        #
        # Contact exists, but the object appears
        # offset from the gripper mid-plane.
        #
        if (
            asymmetry
            > self.config.acceptable_asymmetry
        ):
            return SemanticFeedback(
                state=CONTACT_ASYMMETRIC,
                object_id=object_id,
                message=(
                    "Contact detected but finger "
                    "positions are asymmetric."
                ),
                gripper_positions=q,
                metrics=metrics,
                available_actions=[
                    "RECENTER_GRASP",
                    "REOBSERVE",
                    "CHANGE_GRASP_STRATEGY",
                    "RETURN_READY",
                    "ABORT",
                ],
            )

        #
        # Persistent and sufficiently balanced contact.
        #
        if (
            q1 > min_contact
            and q2 > min_contact
        ):
            return SemanticFeedback(
                state=CONTACT_ACCEPTABLE,
                object_id=object_id,
                message=(
                    "Persistent grasp contact "
                    "is sufficiently balanced."
                ),
                gripper_positions=q,
                metrics=metrics,
                available_actions=[
                    "LIFT_OBJECT",
                    "VERIFY_GRASP",
                    "REOBSERVE",
                ],
            )

        #
        # Contact exists, but only one side clearly
        # reports persistent closure resistance.
        #
        return SemanticFeedback(
            state=CONTACT_DETECTED,
            object_id=object_id,
            message=(
                "Partial contact detected."
            ),
            gripper_positions=q,
            metrics=metrics,
            available_actions=[
                "RECENTER_GRASP",
                "REOBSERVE",
                "RETURN_READY",
                "ABORT",
            ],
        )
