from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import rclpy
from scipy.spatial.transform import Rotation

from physical_ai_runtime.planning.viewpoint_pose import (
    ViewpointPoseConverter,
)


@dataclass(frozen=True)
class ViewpointPlanValidation:
    candidate_id: str

    success: bool
    failure_reason: Optional[str]

    geometric_score: float

    hand_position_world: np.ndarray
    hand_quaternion_world_xyzw: np.ndarray

    hand_position_base: np.ndarray
    hand_quaternion_base_xyzw: np.ndarray

    trajectory: object = None

    trajectory_duration_s: float = float("inf")
    joint_path_length: float = float("inf")
    selection_score: float = 0.0


@dataclass(frozen=True)
class ViewpointPlanSelection:
    results: List[ViewpointPlanValidation]

    selected: Optional[ViewpointPlanValidation]

    safe_candidate_available: bool
    reason: str


class ViewpointPlanValidator:
    """
    Collision-aware PLAN-ONLY validation for viewpoint candidates.

    Safety rules:
      - Uses runtime TF from world -> robot base.
      - Uses adapter.plan_tcp_pose().
      - Adapter safety gate must pass.
      - MoveIt must return a valid collision-aware trajectory.
      - Never executes trajectory.
    """

    def __init__(
        self,
        robot_adapter,
    ):
        self.robot = robot_adapter

        self.pose_converter = (
            ViewpointPoseConverter.from_runtime_tf(
                robot_adapter.tf_buffer,
                robot_adapter.tool_frame,
                getattr(
                    robot_adapter,
                    "camera_optical_frame",
                    None,
                ),
            )
        )

    @staticmethod
    def _transform_to_matrix(tf_msg):
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation

        T = np.eye(4)

        T[:3, :3] = (
            Rotation.from_quat([
                q.x,
                q.y,
                q.z,
                q.w,
            ]).as_matrix()
        )

        T[:3, 3] = [
            t.x,
            t.y,
            t.z,
        ]

        return T

    def _world_hand_to_base(
        self,
        hand_position_world,
        hand_quaternion_world_xyzw,
    ):
        """
        Convert desired hand pose:

            world -> panda_hand

        into:

            panda_link0 -> panda_hand

        using live TF:

            panda_link0 <- world
        """

        try:
            tf = (
                self.robot
                .tf_buffer
                .lookup_transform(
                    self.robot.base_frame,
                    "world",
                    rclpy.time.Time(),
                )
            )

        except Exception as exc:
            return {
                "success": False,
                "failure_reason": (
                    "WORLD_TO_BASE_TF_UNAVAILABLE:"
                    + str(exc)
                ),
            }

        T_base_world = (
            self._transform_to_matrix(
                tf
            )
        )

        T_world_hand = np.eye(4)

        T_world_hand[:3, :3] = (
            Rotation.from_quat(
                hand_quaternion_world_xyzw
            ).as_matrix()
        )

        T_world_hand[:3, 3] = (
            hand_position_world
        )

        T_base_hand = (
            T_base_world
            @ T_world_hand
        )

        hand_position_base = (
            T_base_hand[:3, 3]
        )

        hand_quaternion_base = (
            Rotation.from_matrix(
                T_base_hand[:3, :3]
            ).as_quat()
        )

        return {
            "success": True,
            "position": hand_position_base,
            "orientation":
                hand_quaternion_base,
        }

    def validate_candidate(
        self,
        candidate,
        *,
        timeout=5.0,
    ):
        if not candidate.valid:
            return ViewpointPlanValidation(
                candidate_id=(
                    candidate.candidate_id
                ),
                success=False,
                failure_reason=(
                    "GEOMETRIC_CANDIDATE_INVALID:"
                    + str(candidate.reason)
                ),
                geometric_score=float(
                    candidate.score
                ),
                hand_position_world=(
                    np.full(3, np.nan)
                ),
                hand_quaternion_world_xyzw=(
                    np.full(4, np.nan)
                ),
                hand_position_base=(
                    np.full(3, np.nan)
                ),
                hand_quaternion_base_xyzw=(
                    np.full(4, np.nan)
                ),
                trajectory=None,
                trajectory_duration_s=float("inf"),
                joint_path_length=float("inf"),
                selection_score=0.0,
            )

        pose = self.pose_converter.convert(
            candidate.camera_position_world,
            candidate.target_position_world,
        )

        if not pose.valid:
            return ViewpointPlanValidation(
                candidate_id=(
                    candidate.candidate_id
                ),
                success=False,
                failure_reason=(
                    "POSE_CONVERSION_FAILED:"
                    + str(pose.reason)
                ),
                geometric_score=float(
                    candidate.score
                ),
                hand_position_world=(
                    pose.hand_position_world
                ),
                hand_quaternion_world_xyzw=(
                    pose.hand_quaternion_xyzw
                ),
                hand_position_base=(
                    np.full(3, np.nan)
                ),
                hand_quaternion_base_xyzw=(
                    np.full(4, np.nan)
                ),
                trajectory=None,
                trajectory_duration_s=float("inf"),
                joint_path_length=float("inf"),
                selection_score=0.0,
            )

        base_pose = (
            self._world_hand_to_base(
                pose.hand_position_world,
                pose.hand_quaternion_xyzw,
            )
        )

        if not base_pose["success"]:
            return ViewpointPlanValidation(
                candidate_id=(
                    candidate.candidate_id
                ),
                success=False,
                failure_reason=(
                    base_pose[
                        "failure_reason"
                    ]
                ),
                geometric_score=float(
                    candidate.score
                ),
                hand_position_world=(
                    pose.hand_position_world
                ),
                hand_quaternion_world_xyzw=(
                    pose.hand_quaternion_xyzw
                ),
                hand_position_base=(
                    np.full(3, np.nan)
                ),
                hand_quaternion_base_xyzw=(
                    np.full(4, np.nan)
                ),
                trajectory=None,
                trajectory_duration_s=float("inf"),
                joint_path_length=float("inf"),
                selection_score=0.0,
            )

        plan = self.robot.plan_tcp_pose(
            base_pose["position"],
            base_pose["orientation"],
            timeout=timeout,
        )

        if not plan.get(
            "success",
            False,
        ):
            return ViewpointPlanValidation(
                candidate_id=(
                    candidate.candidate_id
                ),
                success=False,
                failure_reason=(
                    plan.get(
                        "failure_reason",
                        "UNKNOWN_PLANNING_FAILURE",
                    )
                ),
                geometric_score=float(
                    candidate.score
                ),
                hand_position_world=(
                    pose.hand_position_world
                ),
                hand_quaternion_world_xyzw=(
                    pose.hand_quaternion_xyzw
                ),
                hand_position_base=(
                    base_pose["position"]
                ),
                hand_quaternion_base_xyzw=(
                    base_pose["orientation"]
                ),
                trajectory=None,
                trajectory_duration_s=float("inf"),
                joint_path_length=float("inf"),
                selection_score=0.0,
            )

        trajectory = plan["trajectory"]

        points = (
            trajectory
            .joint_trajectory
            .points
        )

        last_time = points[-1].time_from_start

        trajectory_duration_s = float(
            last_time.sec
            + last_time.nanosec / 1e9
        )

        joint_path_length = 0.0

        for previous, current in zip(
            points[:-1],
            points[1:],
        ):
            q0 = np.asarray(
                previous.positions,
                dtype=float,
            )

            q1 = np.asarray(
                current.positions,
                dtype=float,
            )

            joint_path_length += float(
                np.linalg.norm(q1 - q0)
            )

        #
        # View quality remains dominant,
        # but expensive motion receives a penalty.
        #
        # These weights are policy parameters,
        # not physical safety limits.
        #
        selection_score = (
            float(candidate.score)
            - 0.05 * trajectory_duration_s
            - 0.03 * joint_path_length
        )

        return ViewpointPlanValidation(
            candidate_id=(
                candidate.candidate_id
            ),
            success=True,
            failure_reason=None,
            geometric_score=float(
                candidate.score
            ),
            hand_position_world=(
                pose.hand_position_world
            ),
            hand_quaternion_world_xyzw=(
                pose.hand_quaternion_xyzw
            ),
            hand_position_base=(
                base_pose["position"]
            ),
            hand_quaternion_base_xyzw=(
                base_pose["orientation"]
            ),
            trajectory=trajectory,
            trajectory_duration_s=(
                trajectory_duration_s
            ),
            joint_path_length=(
                joint_path_length
            ),
            selection_score=(
                selection_score
            ),
        )

    def validate_candidates(
        self,
        candidates,
        *,
        timeout=5.0,
    ):
        results = []

        for candidate in candidates:
            result = self.validate_candidate(
                candidate,
                timeout=timeout,
            )

            results.append(
                result
            )

        successful = [
            result
            for result in results
            if result.success
        ]

        if not successful:
            return ViewpointPlanSelection(
                results=results,
                selected=None,
                safe_candidate_available=False,
                reason=(
                    "NO_COLLISION_FREE_"
                    "REACHABLE_VIEWPOINT"
                ),
            )

        selected = max(
            successful,
            key=lambda result: (
                result.selection_score
            ),
        )

        return ViewpointPlanSelection(
            results=results,
            selected=selected,
            safe_candidate_available=True,
            reason=(
                "PLAN_ONLY_CANDIDATE_AVAILABLE"
            ),
        )
