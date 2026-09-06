import numpy as np


class RecoveryExecutor:
    """
    Converts a semantic recovery decision into a robot-independent
    recovery request.

    Important:
    - AI chooses WHAT to do.
    - Runtime determines HOW.
    - No scene coordinates are hard-coded here.
    - No direct joint commands are generated here.
    """

    def __init__(self, robot):
        self.robot = robot

    def execute(
        self,
        action,
        feedback,
        grasp_context,
    ):
        name = action.action

        if name == "RECENTER_GRASP":
            return self._build_recenter_request(
                feedback,
                grasp_context,
            )

        if name == "RETRY_GRASP":
            return {
                "success": True,
                "state": "RETRY_GRASP_REQUESTED",
                "object_id": feedback.object_id,
                "strategy": action.strategy,
                "requires_feedback": True,
            }

        if name == "VERIFY_GRASP":
            return {
                "success": True,
                "state": "VERIFY_GRASP_REQUESTED",
                "object_id": feedback.object_id,
                "requires_feedback": True,
            }

        if name == "LIFT_OBJECT":
            return {
                "success": True,
                "state": "LIFT_OBJECT_REQUESTED",
                "object_id": feedback.object_id,
                "requires_feedback": True,
            }

        if name == "RETURN_READY":
            return {
                "success": True,
                "state": "RETURN_READY_REQUESTED",
                "object_id": feedback.object_id,
                "requires_feedback": True,
            }

        if name == "REOBSERVE":
            return {
                "success": True,
                "state": "REOBSERVE_REQUESTED",
                "object_id": feedback.object_id,
            }

        if name == "CHANGE_GRASP_STRATEGY":
            return {
                "success": True,
                "state": "NEW_GRASP_STRATEGY_REQUESTED",
                "object_id": feedback.object_id,
            }

        if name == "ABORT":
            return {
                "success": False,
                "state": "ABORTED",
                "reason": action.reason,
            }

        return {
            "success": False,
            "state": "UNSUPPORTED_ACTION",
            "action": name,
        }

    def _build_recenter_request(
        self,
        feedback,
        grasp_context,
    ):
        metrics = feedback.metrics or {}

        offset = metrics.get(
            "estimated_tool_y_offset"
        )

        if offset is None:
            return {
                "success": False,
                "state": "RECENTER_INPUT_MISSING",
            }

        return {
            "success": True,

            # Semantic command for the runtime controller.
            "state": "RECENTER_REQUESTED",

            "object_id": feedback.object_id,

            # Robot-local axis, not world XYZ.
            "axis": "tool_y",

            # Feedback signal only.
            # Runtime controller decides actual motion step.
            "signed_error": float(offset),

            "orientation": orientation,

            # Runtime must close -> evaluate again after motion.
            "requires_feedback": True,
        }
