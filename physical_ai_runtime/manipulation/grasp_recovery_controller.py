import numpy as np

from physical_ai_runtime.manipulation.grasp_evaluator import GraspEvaluator


class GraspRecoveryController:
    """
    Executes a semantic grasp recovery request.

    Responsibilities:
    - convert tool-local recovery intent into robot motion
    - use adapter capabilities
    - collect new feedback
    - return a new SemanticFeedback

    Non-responsibilities:
    - does not choose the semantic action
    - does not contain scene-specific object coordinates
    - does not command joints directly
    """

    def __init__(
        self,
        robot,
        evaluator=None,
    ):
        self.robot = robot

        self.evaluator = (
            evaluator
            if evaluator is not None
            else GraspEvaluator()
        )

    @staticmethod
    def _quat_to_matrix(q):
        q = np.asarray(q, dtype=float)

        n = np.linalg.norm(q)

        if n == 0.0:
            raise RuntimeError(
                "INVALID_TOOL_ORIENTATION"
            )

        q = q / n

        x, y, z, w = q

        return np.array([
            [
                1 - 2 * (y*y + z*z),
                2 * (x*y - z*w),
                2 * (x*z + y*w),
            ],
            [
                2 * (x*y + z*w),
                1 - 2 * (x*x + z*z),
                2 * (y*z - x*w),
            ],
            [
                2 * (x*z - y*w),
                2 * (y*z + x*w),
                1 - 2 * (x*x + y*y),
            ],
        ], dtype=float)

    def execute_recenter(
        self,
        request,
    ):
        if request.get("state") != "RECENTER_REQUESTED":
            raise RuntimeError(
                "INVALID_RECENTER_REQUEST"
            )

        if request.get("axis") != "tool_y":
            raise RuntimeError(
                "UNSUPPORTED_RECENTER_AXIS"
            )

        object_id = request.get(
            "object_id"
        )

        signed_error = float(
            request["signed_error"]
        )

        #
        # Runtime motion policy.
        #
        # This is not scene geometry.
        # It limits how aggressively any single
        # feedback correction may move the robot.
        #
        max_step = self._get_max_recenter_step()

        correction_scalar = float(
            np.clip(
                -signed_error,
                -max_step,
                max_step,
            )
        )

        tcp_position, tcp_orientation = (
            self.robot.get_tcp_pose()
        )

        tcp_position = np.asarray(
            tcp_position,
            dtype=float,
        )

        tcp_orientation = np.asarray(
            tcp_orientation,
            dtype=float,
        )

        R = self._quat_to_matrix(
            tcp_orientation
        )

        tool_y_axis = R[:, 1]

        correction_vector = (
            tool_y_axis
            * correction_scalar
        )

        corrected_position = (
            tcp_position
            + correction_vector
        )

        print(
            "\nRECENTER CONTROLLER:"
        )
        print(
            "signed_error:",
            signed_error
        )
        print(
            "max_step:",
            max_step
        )
        print(
            "correction_scalar:",
            correction_scalar
        )
        print(
            "tool_y_axis:",
            tool_y_axis
        )
        print(
            "correction_vector:",
            correction_vector
        )
        print(
            "current_tcp:",
            tcp_position
        )
        print(
            "corrected_tcp:",
            corrected_position
        )

        #
        # Re-open enough to release lateral contact.
        #
        open_width = (
            self._get_recovery_open_position()
        )

        reopen = self.robot.move_gripper(
            open_width
        )

        print(
            "\nRECENTER REOPEN:"
        )
        print(reopen)

        reopen_q = np.asarray(
            reopen.actual,
            dtype=float,
        )

        if not self._gripper_clear_enough(
            reopen_q,
            open_width,
        ):
            return self.evaluator.evaluate_contact(
                object_id=object_id,
                gripper_positions=reopen_q,
            )

        #
        # Execute Cartesian recenter motion using
        # the robot adapter's existing planner.
        #
        motion = self.robot.plan_and_execute_tcp(
            float(corrected_position[0]),
            float(corrected_position[1]),
            float(corrected_position[2]),
        )

        print(
            "\nRECENTER MOTION:"
        )
        print(motion)

        if not motion.success:
            return self.evaluator.evaluate_contact(
                object_id=object_id,
                gripper_positions=reopen_q,
            )

        #
        # Close again to obtain new contact feedback.
        #
        close_position = (
            self._get_close_position()
        )

        close = self.robot.move_gripper(
            close_position
        )

        print(
            "\nRECENTER CLOSE:"
        )
        print(close)

        return self.evaluator.evaluate_contact(
            object_id=object_id,
            gripper_positions=close.actual,
        )

    def _get_gripper_limits(self):
        method = getattr(
            self.robot,
            "get_gripper_limits",
            None,
        )

        if not callable(method):
            raise RuntimeError(
                "GRIPPER_LIMIT_API_UNAVAILABLE"
            )

        limits = method()

        if limits is None:
            raise RuntimeError(
                "GRIPPER_LIMITS_UNAVAILABLE"
            )

        return limits

    def _get_recovery_limits(self):
        method = getattr(
            self.robot,
            "get_grasp_recovery_limits",
            None,
        )

        if not callable(method):
            raise RuntimeError(
                "GRASP_RECOVERY_LIMIT_API_UNAVAILABLE"
            )

        limits = method()

        if limits is None:
            raise RuntimeError(
                "GRASP_RECOVERY_LIMITS_UNAVAILABLE"
            )

        return limits

    def _get_max_recenter_step(self):
        limits = self._get_recovery_limits()

        value = limits.get(
            "max_recenter_step"
        )

        if value is None:
            raise RuntimeError(
                "MAX_RECENTER_STEP_UNAVAILABLE"
            )

        return float(value)

    def _get_recovery_open_position(self):
        limits = self._get_gripper_limits()

        return float(
            limits["max_position"]
        )

    def _get_close_position(self):
        limits = self._get_gripper_limits()

        return float(
            limits["min_position"]
        )

    def _gripper_clear_enough(
        self,
        actual,
        commanded_open,
    ):
        if actual.size != 2:
            return False

        limits = self._get_recovery_limits()

        minimum_open_fraction = limits.get(
            "minimum_open_fraction"
        )

        if minimum_open_fraction is None:
            raise RuntimeError(
                "MINIMUM_OPEN_FRACTION_UNAVAILABLE"
            )

        required = (
            float(commanded_open)
            * float(minimum_open_fraction)
        )

        return bool(
            np.min(actual) >= required
        )
