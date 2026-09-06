from dataclasses import dataclass
from typing import Optional

import hashlib
import struct
import time

import numpy as np


@dataclass(frozen=True)
class ExecutionAuthorization:
    authorized: bool
    reason: str

    trajectory_digest: Optional[str]

    start_state_error: float
    trajectory_age_s: float

    safety_context: dict

    authorized_at_monotonic: float


class ExecutionAuthorizationGate:
    """
    Final authorization boundary before robot motion.

    IMPORTANT:
      - does NOT execute motion
      - authorization is bound to exact trajectory contents
      - safety context is rechecked
      - robot start state is rechecked
      - stale trajectories fail closed
    """

    def __init__(
        self,
        robot_adapter,
        *,
        max_start_state_error=0.12,
        max_trajectory_age_s=2.0,
    ):
        self.robot = robot_adapter

        self.max_start_state_error = float(
            max_start_state_error
        )

        self.max_trajectory_age_s = float(
            max_trajectory_age_s
        )

    @staticmethod
    def trajectory_digest(
        robot_trajectory,
    ):
        """
        Stable fingerprint over trajectory joint names,
        positions and timestamps.
        """

        traj = (
            robot_trajectory
            .joint_trajectory
        )

        h = hashlib.sha256()

        for name in traj.joint_names:
            h.update(
                name.encode("utf-8")
            )
            h.update(b"\0")

        for point in traj.points:

            h.update(
                struct.pack(
                    "<qI",
                    int(
                        point
                        .time_from_start
                        .sec
                    ),
                    int(
                        point
                        .time_from_start
                        .nanosec
                    ),
                )
            )

            for value in point.positions:
                h.update(
                    struct.pack(
                        "<d",
                        float(value),
                    )
                )

        return h.hexdigest()

    def _deny(
        self,
        reason,
        *,
        digest=None,
        start_state_error=float("inf"),
        trajectory_age_s=float("inf"),
        safety_context=None,
    ):
        return ExecutionAuthorization(
            authorized=False,
            reason=reason,
            trajectory_digest=digest,
            start_state_error=(
                start_state_error
            ),
            trajectory_age_s=(
                trajectory_age_s
            ),
            safety_context=(
                safety_context or {}
            ),
            authorized_at_monotonic=(
                time.monotonic()
            ),
        )

    def authorize(
        self,
        robot_trajectory,
        *,
        planned_at_monotonic,
    ):
        if robot_trajectory is None:
            return self._deny(
                "TRAJECTORY_MISSING"
            )

        traj = (
            robot_trajectory
            .joint_trajectory
        )

        if not traj.points:
            return self._deny(
                "TRAJECTORY_EMPTY"
            )

        digest = self.trajectory_digest(
            robot_trajectory
        )

        if planned_at_monotonic is None:
            return self._deny(
                "TRAJECTORY_TIMESTAMP_MISSING",
                digest=digest,
            )

        age = (
            time.monotonic()
            - float(
                planned_at_monotonic
            )
        )

        if (
            age < 0.0
            or age
            > self.max_trajectory_age_s
        ):
            return self._deny(
                "TRAJECTORY_STALE",
                digest=digest,
                trajectory_age_s=age,
            )

        safety = (
            self.robot
            .verify_motion_safety_context()
        )

        if not safety.get(
            "success",
            False,
        ):
            return self._deny(
                (
                    "SAFETY_CONTEXT_FAILED:"
                    + str(
                        safety.get(
                            "failure_reason",
                            "UNKNOWN",
                        )
                    )
                ),
                digest=digest,
                trajectory_age_s=age,
                safety_context=safety,
            )

        self.robot.wait_for_joint_state()

        index = {
            name: i
            for i, name in enumerate(
                traj.joint_names
            )
        }

        try:
            planned_start = np.array(
                [
                    traj.points[0]
                    .positions[
                        index[name]
                    ]
                    for name
                    in self.robot.joint_names
                ],
                dtype=float,
            )

        except KeyError:
            return self._deny(
                "TRAJECTORY_JOINT_MISMATCH",
                digest=digest,
                trajectory_age_s=age,
                safety_context=safety,
            )

        actual = np.asarray(
            self.robot
            .current_joint_position,
            dtype=float,
        )

        if (
            actual.shape
            != planned_start.shape
        ):
            return self._deny(
                "START_STATE_SHAPE_MISMATCH",
                digest=digest,
                trajectory_age_s=age,
                safety_context=safety,
            )

        error = float(
            np.linalg.norm(
                actual
                - planned_start
            )
        )

        if (
            error
            > self.max_start_state_error
        ):
            return self._deny(
                "START_STATE_CHANGED",
                digest=digest,
                start_state_error=error,
                trajectory_age_s=age,
                safety_context=safety,
            )

        return ExecutionAuthorization(
            authorized=True,
            reason="EXECUTION_AUTHORIZED",
            trajectory_digest=digest,
            start_state_error=error,
            trajectory_age_s=age,
            safety_context=safety,
            authorized_at_monotonic=(
                time.monotonic()
            ),
        )

    def verify_authorization(
        self,
        robot_trajectory,
        authorization,
        *,
        max_authorization_age_s=0.5,
    ):
        """
        Final check immediately before execute.

        No motion is performed here.
        """

        if (
            authorization is None
            or not authorization.authorized
        ):
            return False, (
                "AUTHORIZATION_MISSING_OR_DENIED"
            )

        current_digest = (
            self.trajectory_digest(
                robot_trajectory
            )
        )

        if (
            current_digest
            != authorization
            .trajectory_digest
        ):
            return False, (
                "TRAJECTORY_CHANGED_AFTER_AUTHORIZATION"
            )

        authorization_age = (
            time.monotonic()
            - authorization
            .authorized_at_monotonic
        )

        if (
            authorization_age < 0.0
            or authorization_age
            > float(
                max_authorization_age_s
            )
        ):
            return False, (
                "AUTHORIZATION_STALE"
            )

        safety = (
            self.robot
            .verify_motion_safety_context()
        )

        if not safety.get(
            "success",
            False,
        ):
            return False, (
                "SAFETY_CONTEXT_CHANGED"
            )

        return True, (
            "AUTHORIZED_TRAJECTORY_VERIFIED"
        )
