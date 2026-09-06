import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from moveit_msgs.srv import GetPositionIK
from tf2_ros import Buffer, TransformListener

from physical_ai_runtime.core.base_adapter import BaseRobotAdapter
from physical_ai_runtime.core.robot_capabilities import RobotCapabilities
from physical_ai_runtime.core.robot_types import RobotState, SkillResult


class GazeboUR5eAdapter(BaseRobotAdapter, Node):

    def __init__(self):
        Node.__init__(
            self,
            "gazebo_ur5e_runtime_adapter"
        )

        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        self.current_joint_position = None
        self.current_joint_velocity = None

        # =========================
        # Joint state subscriber
        # =========================

        self.create_subscription(
            JointState,
            "/ur5e/joint_states",
            self._joint_state_callback,
            10
        )

        # =========================
        # Trajectory action
        # =========================

        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/ur5e/scaled_joint_trajectory_controller/follow_joint_trajectory"
        )

        # =========================
        # MoveIt IK
        # =========================

        self.ik_client = self.create_client(
            GetPositionIK,
            "/ur5e/compute_ik"
        )

        # =========================
        # TF
        # =========================

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        self.base_frame = "base_link"
        self.tool_frame = "tool0"
        self.move_group = "ur_manipulator"

        self.robot_name = "ur5e"

    # ==========================================================
    # CAPABILITIES
    # ==========================================================

    def get_capabilities(self):
        return RobotCapabilities(
            move_tcp=True,
            grasp=False,
            force_control=False,
            locomotion=False,
            move_joints=True
        )

    # ==========================================================
    # JOINT STATE
    # ==========================================================

    def _joint_state_callback(self, msg):

        pos_map = dict(
            zip(
                msg.name,
                msg.position
            )
        )

        if len(msg.velocity) == len(msg.name):
            velocities = msg.velocity
        else:
            velocities = [0.0] * len(msg.name)

        vel_map = dict(
            zip(
                msg.name,
                velocities
            )
        )

        try:
            self.current_joint_position = np.array([
                pos_map[name]
                for name in self.joint_names
            ])

            self.current_joint_velocity = np.array([
                vel_map[name]
                for name in self.joint_names
            ])

        except KeyError:
            return

    def wait_for_joint_state(self, timeout=5.0):

        start = time.time()

        while (
            rclpy.ok()
            and self.current_joint_position is None
        ):

            rclpy.spin_once(
                self,
                timeout_sec=0.1
            )

            if time.time() - start > timeout:
                raise RuntimeError(
                    "JOINT_STATE_TIMEOUT"
                )

    def get_state(self):

        self.wait_for_joint_state()

        tcp_position, _ = self.get_tcp_pose()

        return RobotState(
            joint_position=
                self.current_joint_position.copy(),

            joint_velocity=
                self.current_joint_velocity.copy(),

            tcp_position=
                tcp_position.copy()
        )

    # ==========================================================
    # TCP POSE
    # ==========================================================

    def get_tcp_pose(self, timeout=5.0):

        start = time.time()

        while rclpy.ok():

            try:
                transform = (
                    self.tf_buffer.lookup_transform(
                        self.base_frame,
                        self.tool_frame,
                        rclpy.time.Time()
                    )
                )

                t = transform.transform.translation
                r = transform.transform.rotation

                position = np.array([
                    t.x,
                    t.y,
                    t.z
                ])

                orientation = np.array([
                    r.x,
                    r.y,
                    r.z,
                    r.w
                ])

                return (
                    position,
                    orientation
                )

            except Exception:

                rclpy.spin_once(
                    self,
                    timeout_sec=0.1
                )

            if time.time() - start > timeout:
                raise RuntimeError(
                    "TCP_TRANSFORM_TIMEOUT"
                )

    # ==========================================================
    # MOVEIT IK
    # ==========================================================

    def solve_ik(
        self,
        x,
        y,
        z,
        timeout=5.0
    ):

        self.wait_for_joint_state()

        if not self.ik_client.wait_for_service(
            timeout_sec=timeout
        ):
            raise RuntimeError(
                "IK_SERVICE_NOT_AVAILABLE"
            )

        _, current_orientation = self.get_tcp_pose()

        # First try current configuration.
        # If the robot starts near a difficult / singular pose,
        # retry using a bent nominal UR5e configuration.
        seeds = [
            self.current_joint_position.copy(),

            np.array([
                0.0,
                -1.2,
                1.4,
                -1.5,
                -1.57,
                0.0
            ], dtype=float)
        ]

        last_error = None

        for seed in seeds:

            request = GetPositionIK.Request()

            request.ik_request.group_name = self.move_group
            request.ik_request.ik_link_name = self.tool_frame
            request.ik_request.avoid_collisions = False

            request.ik_request.robot_state.joint_state.name = (
                self.joint_names
            )

            request.ik_request.robot_state.joint_state.position = (
                seed.tolist()
            )

            request.ik_request.pose_stamped.header.frame_id = (
                self.base_frame
            )

            request.ik_request.pose_stamped.pose.position.x = float(x)
            request.ik_request.pose_stamped.pose.position.y = float(y)
            request.ik_request.pose_stamped.pose.position.z = float(z)

            request.ik_request.pose_stamped.pose.orientation.x = float(
                current_orientation[0]
            )
            request.ik_request.pose_stamped.pose.orientation.y = float(
                current_orientation[1]
            )
            request.ik_request.pose_stamped.pose.orientation.z = float(
                current_orientation[2]
            )
            request.ik_request.pose_stamped.pose.orientation.w = float(
                current_orientation[3]
            )

            future = self.ik_client.call_async(request)

            rclpy.spin_until_future_complete(
                self,
                future
            )

            response = future.result()

            if response is None:
                last_error = "IK_SERVICE_FAILED"
                continue

            if response.error_code.val != 1:
                last_error = (
                    f"IK_NOT_CONVERGED:"
                    f"{response.error_code.val}"
                )
                continue

            solution = response.solution.joint_state

            joint_map = dict(
                zip(
                    solution.name,
                    solution.position
                )
            )

            try:
                return np.array([
                    joint_map[name]
                    for name in self.joint_names
                ])

            except KeyError as e:
                last_error = (
                    f"IK_SOLUTION_MISSING_JOINT:{e}"
                )

        raise RuntimeError(
            last_error or "IK_NOT_CONVERGED"
        )

    # ==========================================================
    # STOP
    # ==========================================================

    def stop(self):

        self.wait_for_joint_state()

        return self.move_joints(
            self.current_joint_position,
            duration=0.5
        )

    # ==========================================================
    # JOINT TRAJECTORY
    # ==========================================================

    def move_joints(
        self,
        target_q,
        duration=3.0
    ):

        target_q = np.array(
            target_q,
            dtype=float
        )

        self.wait_for_joint_state()

        if not self.trajectory_client.wait_for_server(
            timeout_sec=5.0
        ):
            raise RuntimeError(
                "TRAJECTORY_ACTION_NOT_AVAILABLE"
            )

        goal = FollowJointTrajectory.Goal()

        goal.trajectory.joint_names = (
            self.joint_names
        )

        point = JointTrajectoryPoint()

        point.positions = (
            target_q.tolist()
        )

        point.time_from_start.sec = int(
            duration
        )

        point.time_from_start.nanosec = int(
            (duration - int(duration))
            * 1e9
        )

        goal.trajectory.points = [
            point
        ]

        future = (
            self.trajectory_client
            .send_goal_async(goal)
        )

        rclpy.spin_until_future_complete(
            self,
            future
        )

        goal_handle = future.result()

        if not goal_handle.accepted:
            return SkillResult(
                skill="MOVE_JOINTS",
                success=False,
                target=target_q,
                actual=
                    self.current_joint_position.copy(),
                error=0.0,
                duration=0.0,
                failure_reason=
                    "TRAJECTORY_GOAL_REJECTED"
            )

        start = time.time()

        result_future = (
            goal_handle.get_result_async()
        )

        rclpy.spin_until_future_complete(
            self,
            result_future
        )

        result = (
            result_future
            .result()
            .result
        )

        rclpy.spin_once(
            self,
            timeout_sec=0.1
        )

        actual = (
            self.current_joint_position.copy()
        )

        error = float(
            np.linalg.norm(
                target_q - actual
            )
        )

        return SkillResult(
            skill="MOVE_JOINTS",
            success=(
                result.error_code == 0
            ),
            target=target_q,
            actual=actual,
            error=error,
            duration=(
                time.time() - start
            ),
            failure_reason=(
                None
                if result.error_code == 0
                else result.error_string
            )
        )

    # ==========================================================
    # MOVE TCP
    # ==========================================================

    def execute_move(
        self,
        x,
        y,
        z,
        tolerance=0.01,
        timeout=5.0,
        duration=3.0
    ):

        target_xyz = np.array(
            [x, y, z],
            dtype=float
        )

        start = time.time()

        # -------------------------
        # IK
        # -------------------------

        try:
            target_q = self.solve_ik(
                x,
                y,
                z,
                timeout=timeout
            )

        except RuntimeError as e:

            return SkillResult(
                skill="MOVE_TCP",
                success=False,
                target=target_xyz,
                actual=np.array([]),
                error=0.0,
                duration=(
                    time.time() - start
                ),
                failure_reason=str(e)
            )

        # -------------------------
        # Execute trajectory
        # -------------------------

        trajectory_result = (
            self.move_joints(
                target_q,
                duration=duration
            )
        )

        if not trajectory_result.success:

            return SkillResult(
                skill="MOVE_TCP",
                success=False,
                target=target_xyz,
                actual=np.array([]),
                error=0.0,
                duration=(
                    time.time() - start
                ),
                failure_reason=(
                    trajectory_result
                    .failure_reason
                )
            )

        # -------------------------
        # Verify TCP
        # -------------------------

        verify_start = time.time()

        actual_xyz = np.array([])
        error = float("inf")

        while rclpy.ok():

            try:
                actual_xyz, _ = self.get_tcp_pose(
                    timeout=0.5
                )

            except RuntimeError:
                actual_xyz = np.array([])

            if actual_xyz.size == 3:

                error = float(
                    np.linalg.norm(
                        target_xyz
                        - actual_xyz
                    )
                )

                if error <= tolerance:
                    break

            if time.time() - verify_start > timeout:
                break

            rclpy.spin_once(
                self,
                timeout_sec=0.05
            )

        success = (
            actual_xyz.size == 3
            and error <= tolerance
        )

        return SkillResult(
            skill="MOVE_TCP",
            success=success,
            target=target_xyz,
            actual=actual_xyz,
            error=(
                error
                if np.isfinite(error)
                else 0.0
            ),
            duration=(
                time.time() - start
            ),
            failure_reason=(
                None
                if success
                else "TARGET_NOT_REACHED"
            )
        )