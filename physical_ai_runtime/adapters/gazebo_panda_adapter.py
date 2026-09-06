import time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

from moveit_msgs.srv import (
    GetPositionIK,
    GetMotionPlan,
    GetPlanningScene,
)
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
)
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from tf2_ros import Buffer, TransformListener

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import ExecuteTrajectory

from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray

from physical_ai_runtime.core.base_adapter import BaseRobotAdapter
from physical_ai_runtime.core.robot_capabilities import RobotCapabilities
from physical_ai_runtime.core.robot_types import RobotState, SkillResult
from physical_ai_runtime.planning.execution_authorization import (
    ExecutionAuthorizationGate,
)


class GazeboPandaAdapter(BaseRobotAdapter, Node):

    def __init__(self):
        Node.__init__(
            self,
            "gazebo_panda_runtime_adapter"
        )

        self.robot_name = "panda"

        self.joint_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]

        self.current_joint_position = None
        self.current_joint_velocity = None
        self.current_joint_effort = None
        self.joint_effort_available = False
        self.joint_effort_timestamp = None

        self.current_gripper_position = None

        self.gripper_pub = self.create_publisher(
            JointTrajectory,
            "/panda_gripper_native_trajectory",
            10
        )

        self.create_subscription(
            JointState,
            "/panda_gripper_joint_states",
            self._gripper_state_callback,
            10
        )

        self.create_subscription(
            JointState,
            "/panda/joint_states",
            self._joint_state_callback,
            10
        )

        self.position_pub = self.create_publisher(
            Float64MultiArray,
            "/panda/panda_arm_controller/commands",
            10
        )

        self.ik_client = self.create_client(
            GetPositionIK,
            "/panda/compute_ik"
        )


        self.motion_plan_client = self.create_client(
            GetMotionPlan,
            "/panda/plan_kinematic_path"
        )

        self.planning_scene_client = self.create_client(
            GetPlanningScene,
            "/panda/get_planning_scene"
        )

        self.execute_trajectory_client = ActionClient(
            self,
            ExecuteTrajectory,
            "/panda/execute_trajectory"
        )

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        self.base_frame = "panda_link0"
        self.tool_frame = "panda_hand"
        self.camera_optical_frame = (
            "panda_wrist_camera_optical_frame"
        )
        self.move_group = "panda_arm"

    def preflight_check(
        self,
        timeout=3.0
    ):
        """
        Verify that the robot/runtime is safe to enter
        autonomous motion.
        """

        checks = {}

        #
        # Joint state
        #
        try:
            self.wait_for_joint_state(
                timeout=timeout
            )
            checks["joint_state"] = True
        except Exception:
            checks["joint_state"] = False

        #
        # Gripper state
        #
        deadline = time.time() + timeout

        while (
            self.current_gripper_position is None
            and time.time() < deadline
        ):
            rclpy.spin_once(
                self,
                timeout_sec=0.1
            )

        checks["gripper_state"] = (
            self.current_gripper_position
            is not None
        )

        #
        # Motion planning service
        #
        checks["motion_planner"] = (
            self.motion_plan_client.wait_for_service(
                timeout_sec=timeout
            )
        )

        #
        # Planning scene safety context
        #
        safety = (
            self.verify_motion_safety_context(
                timeout=timeout
            )
        )

        checks["planning_scene"] = bool(
            safety["success"]
        )

        success = all(
            checks.values()
        )

        return {
            "success": success,
            "checks": checks,
            "safety_context": safety,
            "failure_reason": (
                None
                if success
                else "PREFLIGHT_FAILED"
            ),
        }


    def verify_motion_safety_context(
        self,
        required_collision_ids=None,
        timeout=3.0
    ):
        """
        Verify that the MoveIt PlanningScene contains the
        collision objects required before autonomous arm motion.
        """

        if required_collision_ids is None:
            required_collision_ids = [
                "detected_support_surface"
            ]

        if not self.planning_scene_client.wait_for_service(
            timeout_sec=timeout
        ):
            return {
                "success": False,
                "failure_reason":
                    "PLANNING_SCENE_SERVICE_UNAVAILABLE",
                "present_collision_ids": [],
            }

        request = GetPlanningScene.Request()

        # Request full scene components.
        request.components.components = 1023

        future = self.planning_scene_client.call_async(
            request
        )

        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=timeout
        )

        if not future.done():
            return {
                "success": False,
                "failure_reason":
                    "PLANNING_SCENE_TIMEOUT",
                "present_collision_ids": [],
            }

        response = future.result()

        if response is None:
            return {
                "success": False,
                "failure_reason":
                    "PLANNING_SCENE_UNAVAILABLE",
                "present_collision_ids": [],
            }

        present = {
            obj.id
            for obj in (
                response.scene.world
                .collision_objects
            )
        }

        missing = [
            object_id
            for object_id
            in required_collision_ids
            if object_id not in present
        ]

        if missing:
            return {
                "success": False,
                "failure_reason":
                    "SAFETY_CONTEXT_INCOMPLETE",
                "missing_collision_ids":
                    missing,
                "present_collision_ids":
                    sorted(present),
            }

        return {
            "success": True,
            "failure_reason": None,
            "present_collision_ids":
                sorted(present),
        }


    def get_capabilities(self):
        return RobotCapabilities(
            move_tcp=True,
            grasp=True,
            force_control=False,
            locomotion=False,
            move_joints=True
        )


    def get_ready_joint_positions(self):
        """
        Read the configured ready/initial arm posture from
        the Panda xacro source of truth.

        No joint values are duplicated in runtime code.
        """

        xacro_path = (
            Path(__file__).resolve().parents[2]
            / "panda_gz"
            / "panda_gazebo.urdf.xacro"
        )

        if not xacro_path.exists():
            raise RuntimeError(
                f"ROBOT_DESCRIPTION_NOT_FOUND: {xacro_path}"
            )

        root = ET.parse(
            xacro_path
        ).getroot()

        ros2_control = root.find(
            "./ros2_control"
        )

        if ros2_control is None:
            raise RuntimeError(
                "ROS2_CONTROL_DESCRIPTION_NOT_FOUND"
            )

        positions = []

        for joint_name in self.joint_names:

            joint = ros2_control.find(
                f"./joint[@name='{joint_name}']"
            )

            if joint is None:
                raise RuntimeError(
                    f"CONTROL_JOINT_NOT_FOUND: {joint_name}"
                )

            initial_value = None

            #
            # Search both command/state interfaces because
            # different ros2_control descriptions may place
            # initial_value differently.
            #
            for interface in list(joint):
                param = interface.find(
                    "./param[@name='initial_value']"
                )

                if param is not None:
                    initial_value = float(
                        param.text
                    )
                    break

            if initial_value is None:
                raise RuntimeError(
                    "READY_POSITION_UNAVAILABLE: "
                    f"{joint_name}"
                )

            positions.append(
                initial_value
            )

        return np.asarray(
            positions,
            dtype=float
        )


    def return_to_ready(self):
        """
        Return the robot to its configured ready posture.

        The joint target is read from the robot description,
        not duplicated in runtime code.
        """

        target = self.get_ready_joint_positions()

        return self.plan_and_execute_joints(
            target
        )

    def get_gripper_limits(self):
        """
        Read gripper joint limits directly from the robot URDF.

        No Panda finger travel values are duplicated in runtime logic.
        """

        urdf_path = (
            Path(__file__).resolve().parents[2]
            / "panda_gz"
            / "panda_gazebo.urdf"
        )

        if not urdf_path.exists():
            raise RuntimeError(
                f"ROBOT_DESCRIPTION_NOT_FOUND: {urdf_path}"
            )

        root = ET.parse(
            urdf_path
        ).getroot()

        joints = []

        for joint_name in (
            "panda_finger_joint1",
            "panda_finger_joint2",
        ):
            joint = root.find(
                f"./joint[@name='{joint_name}']"
            )

            if joint is None:
                raise RuntimeError(
                    f"GRIPPER_JOINT_NOT_FOUND: {joint_name}"
                )

            limit = joint.find("limit")

            if limit is None:
                raise RuntimeError(
                    f"GRIPPER_LIMIT_NOT_FOUND: {joint_name}"
                )

            axis = joint.find("axis")

            joints.append({
                "name": joint_name,
                "lower": float(
                    limit.attrib["lower"]
                ),
                "upper": float(
                    limit.attrib["upper"]
                ),
                "velocity": float(
                    limit.attrib["velocity"]
                ),
                "effort": float(
                    limit.attrib["effort"]
                ),
                "axis": (
                    None
                    if axis is None
                    else np.asarray(
                        [
                            float(v)
                            for v
                            in axis.attrib["xyz"].split()
                        ],
                        dtype=float,
                    )
                ),
            })

        lower = max(
            j["lower"]
            for j in joints
        )

        upper = min(
            j["upper"]
            for j in joints
        )

        return {
            "min_position": lower,
            "max_position": upper,
            "travel": upper - lower,
            "joints": joints,
        }


    def get_grasp_recovery_limits(self):
        """
        Return robot-specific recovery policy derived from
        gripper capabilities.

        No scene/object-specific values are used here.
        """

        gripper = self.get_gripper_limits()

        travel = float(
            gripper["travel"]
        )

        if travel <= 0.0:
            raise RuntimeError(
                "INVALID_GRIPPER_TRAVEL"
            )

        return {
            # Maximum Cartesian recenter motion allowed
            # per feedback iteration.
            "max_recenter_step": (
                0.10 * travel
            ),

            # Required fraction of maximum finger opening
            # before lateral recenter motion is allowed.
            "minimum_open_fraction": 0.85,

            # Number of semantic recovery attempts exposed
            # to higher-level policy.
            "max_recenter_attempts": 4,
        }

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

        effort_valid = (
            len(msg.effort) == len(msg.name)
        )

        if effort_valid:
            effort_map = dict(
                zip(
                    msg.name,
                    msg.effort
                )
            )
        else:
            effort_map = {}

        try:
            self.current_joint_position = np.array([
                pos_map[name]
                for name in self.joint_names
            ])

            self.current_joint_velocity = np.array([
                vel_map[name]
                for name in self.joint_names
            ])

            if effort_valid:
                self.current_joint_effort = np.array([
                    effort_map[name]
                    for name in self.joint_names
                ])

                self.joint_effort_available = True
                self.joint_effort_timestamp = (
                    self.get_clock().now().nanoseconds
                    * 1e-9
                )
            else:
                self.current_joint_effort = None
                self.joint_effort_available = False
                self.joint_effort_timestamp = None

        except KeyError:
            return

    def _gripper_state_callback(self, msg):

        pos_map = dict(zip(msg.name, msg.position))

        try:
            self.current_gripper_position = np.array([
                pos_map["panda_finger_joint1"],
                pos_map["panda_finger_joint2"],
            ])
        except KeyError:
            return

    def wait_for_joint_state(
        self,
        timeout=5.0
    ):

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

        return RobotState(
            joint_position=
                self.current_joint_position.copy(),

            joint_velocity=
                self.current_joint_velocity.copy(),

            tcp_position=np.array([
                np.nan,
                np.nan,
                np.nan
            ])
        )

    def stop(self):
        self.wait_for_joint_state()

        hold_position = self.current_joint_position.copy()

        command = Float64MultiArray()
        command.data = hold_position.tolist()
        self.position_pub.publish(command)

        return SkillResult(
            skill="STOP_HOLD",
            success=True,
            target=hold_position,
            actual=hold_position,
            error=0.0,
            duration=0.0,
            failure_reason=None,
        )

    def move_joints(
        self,
        target_q,
        duration=5.0
    ):

        target_q = np.array(
            target_q,
            dtype=float
        )

        self.wait_for_joint_state()

        start_q = self.current_joint_position.copy()

        start = time.time()

        # Smooth position interpolation instead of
        # jumping directly to the final joint target.
        while rclpy.ok():

            elapsed = time.time() - start

            alpha = min(
                elapsed / max(duration, 0.001),
                1.0
            )

            command_q = (
                start_q
                + alpha * (target_q - start_q)
            )

            msg = Float64MultiArray()
            msg.data = command_q.tolist()

            self.position_pub.publish(msg)

            rclpy.spin_once(
                self,
                timeout_sec=0.02
            )

            if alpha >= 1.0:
                break

        # Keep publishing final target while waiting
        # for feedback to settle.
        settle_start = time.time()

        actual = self.current_joint_position.copy()
        error = float("inf")

        final_msg = Float64MultiArray()
        final_msg.data = target_q.tolist()

        while rclpy.ok():

            self.position_pub.publish(final_msg)

            rclpy.spin_once(
                self,
                timeout_sec=0.05
            )

            actual = self.current_joint_position.copy()

            error = float(
                np.max(
                    np.abs(
                        target_q - actual
                    )
                )
            )

            if error <= 0.01:
                break

            if time.time() - settle_start > 5.0:
                break

        success = error <= 0.01

        return SkillResult(
            skill="MOVE_JOINTS",
            success=success,
            target=target_q,
            actual=actual,
            error=error,
            duration=time.time() - start,
            failure_reason=(
                None
                if success
                else "JOINT_TARGET_NOT_REACHED"
            )
        )


    def go_home(
        self,
        duration=3.0
    ):
        home_q = np.array([
            0.0,
            -0.785,
            0.0,
            -2.356,
            0.0,
            1.571,
            0.785
        ])

        return self.move_joints(
            home_q,
            duration=duration
        )

    def move_gripper(
        self,
        width,
        speed=0.03,
        tolerance=0.002,
        timeout=3.0
    ):

        target = float(np.clip(width, 0.0, 0.04))
        target_array = np.array([target, target])

        start = time.time()

        command_deadline = time.time() + 5.0

        while (
            rclpy.ok()
            and self.gripper_pub.get_subscription_count() == 0
        ):
            rclpy.spin_once(self, timeout_sec=0.1)

            if time.time() >= command_deadline:
                return SkillResult(
                    skill="GRIPPER",
                    success=False,
                    target=target_array,
                    actual=(
                        np.array([])
                        if self.current_gripper_position is None
                        else self.current_gripper_position.copy()
                    ),
                    error=0.0,
                    duration=time.time() - start,
                    failure_reason="GRIPPER_CONTROLLER_NOT_CONNECTED"
                )

        state_deadline = time.time() + 5.0

        while (
            rclpy.ok()
            and self.current_gripper_position is None
        ):
            rclpy.spin_once(self, timeout_sec=0.1)

            if time.time() >= state_deadline:
                return SkillResult(
                    skill="GRIPPER",
                    success=False,
                    target=target_array,
                    actual=np.array([]),
                    error=0.0,
                    duration=time.time() - start,
                    failure_reason="GRIPPER_STATE_TIMEOUT"
                )

        msg = JointTrajectory()
        msg.joint_names = [
            "panda_finger_joint1",
            "panda_finger_joint2",
        ]

        point = JointTrajectoryPoint()
        point.positions = [target, target]
        point.time_from_start.sec = 1
        point.time_from_start.nanosec = 0

        msg.points = [point]

        self.gripper_pub.publish(msg)

        motion_start = time.time()

        actual = self.current_gripper_position.copy()
        error = float(np.max(np.abs(actual - target_array)))

        while rclpy.ok():

            rclpy.spin_once(self, timeout_sec=0.02)

            if self.current_gripper_position is not None:
                actual = self.current_gripper_position.copy()

                error = float(
                    np.max(
                        np.abs(actual - target_array)
                    )
                )

                if error <= tolerance:
                    break

            if time.time() - motion_start >= timeout:
                break

        success = error <= tolerance

        return SkillResult(
            skill="GRIPPER",
            success=success,
            target=target_array,
            actual=actual,
            error=error,
            duration=time.time() - start,
            failure_reason=(
                None
                if success
                else "GRIPPER_TARGET_NOT_REACHED"
            )
        )

    def get_grasp_tool_envelope(self):
        """
        Grasping geometry measured along panda_hand local +Z.

        near_offset:
            start of finger collision geometry.

        far_offset:
            distal end of finger collision geometry.

        support_clearance:
            minimum clearance between fingertip and support surface.
        """
        finger_joint_z = 0.0584

        finger_min_z = 0.00013169624435249716
        finger_max_z = 0.05384903401136398

        return {
            "near_offset": (
                finger_joint_z + finger_min_z
            ),
            "far_offset": (
                finger_joint_z + finger_max_z
            ),
            "support_clearance": 0.002,
        }

    def get_grasp_tool_offset(self) -> float:
        """
        Nominal distance from panda_hand origin to the grasp contact
        region along hand +Z.

        The finger collision geometry spans approximately:
        z = 0.000132 .. 0.053849 m
        relative to the finger joint origin.

        We use the center of that span as the nominal side-contact
        region rather than the fingertip.
        """
        finger_joint_z = 0.0584
        finger_min_z = 0.00013169624435249716
        finger_max_z = 0.05384903401136398

        return (
            finger_joint_z
            + 0.5 * (finger_min_z + finger_max_z)
        )

    def get_tcp_pose(
        self,
        timeout=5.0
    ):

        start = time.time()

        while rclpy.ok():

            try:
                transform = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    self.tool_frame,
                    rclpy.time.Time()
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

                return position, orientation

            except Exception:

                rclpy.spin_once(
                    self,
                    timeout_sec=0.1
                )

            if time.time() - start > timeout:
                raise RuntimeError(
                    "TCP_TRANSFORM_TIMEOUT"
                )

    def plan_and_execute_lift(
        self,
        distance,
        tolerance=0.01,
        timeout=10.0,
    ):
        """
        Collision-aware Cartesian lift along world +Z.

        The requested distance is semantic/runtime input.
        World-up is transformed into the robot base frame from TF.
        """

        distance = float(distance)

        if (
            not np.isfinite(distance)
            or distance <= 0.0
        ):
            return SkillResult(
                skill="LIFT_OBJECT",
                success=False,
                target=None,
                actual=None,
                error=None,
                duration=0.0,
                failure_reason="INVALID_LIFT_DISTANCE",
            )

        safety = self.verify_motion_safety_context(
            timeout=3.0
        )

        if not safety["success"]:
            return SkillResult(
                skill="LIFT_OBJECT",
                success=False,
                target=None,
                actual=None,
                error=None,
                duration=0.0,
                failure_reason=(
                    safety.get(
                        "failure_reason",
                        "SAFETY_CONTEXT_UNAVAILABLE",
                    )
                ),
            )

        start_time = time.time()

        try:
            current_position, current_orientation = (
                self.get_tcp_pose(
                    timeout=timeout
                )
            )

            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                "world",
                rclpy.time.Time(),
            )

            q = tf.transform.rotation

            x = q.x
            y = q.y
            z = q.z
            w = q.w

            rotation = np.array([
                [
                    1.0 - 2.0 * (y*y + z*z),
                    2.0 * (x*y - z*w),
                    2.0 * (x*z + y*w),
                ],
                [
                    2.0 * (x*y + z*w),
                    1.0 - 2.0 * (x*x + z*z),
                    2.0 * (y*z - x*w),
                ],
                [
                    2.0 * (x*z - y*w),
                    2.0 * (y*z + x*w),
                    1.0 - 2.0 * (x*x + y*y),
                ],
            ], dtype=float)

            world_up_in_base = (
                rotation
                @ np.array(
                    [0.0, 0.0, 1.0],
                    dtype=float,
                )
            )

            norm = float(
                np.linalg.norm(
                    world_up_in_base
                )
            )

            if norm <= 1e-9:
                raise RuntimeError(
                    "WORLD_UP_TRANSFORM_INVALID"
                )

            world_up_in_base /= norm

            target = (
                np.asarray(
                    current_position,
                    dtype=float,
                )
                + world_up_in_base * distance
            )

            plan = self.plan_tcp_pose(
                target=target,
                orientation=np.asarray(
                    current_orientation,
                    dtype=float,
                ),
                timeout=timeout,
            )

            if not plan["success"]:
                return SkillResult(
                    skill="LIFT_OBJECT",
                    success=False,
                    target=target,
                    actual=current_position,
                    error=None,
                    duration=time.time() - start_time,
                    failure_reason=(
                        plan.get(
                            "failure_reason",
                            "LIFT_PLANNING_FAILED",
                        )
                    ),
                )

            execution = (
                self.execute_planned_trajectory(
                    plan["trajectory"]
                )
            )

            if not execution.success:
                return SkillResult(
                    skill="LIFT_OBJECT",
                    success=False,
                    target=target,
                    actual=None,
                    error=None,
                    duration=time.time() - start_time,
                    failure_reason=(
                        execution.failure_reason
                    ),
                )

            actual_position, _ = self.get_tcp_pose(
                timeout=timeout
            )

            actual_position = np.asarray(
                actual_position,
                dtype=float,
            )

            achieved = float(
                np.dot(
                    actual_position
                    - current_position,
                    world_up_in_base,
                )
            )

            error = abs(
                distance - achieved
            )

            success = (
                achieved > 0.0
                and error <= float(tolerance)
            )

            return SkillResult(
                skill="LIFT_OBJECT",
                success=success,
                target=target,
                actual=actual_position,
                error=error,
                duration=time.time() - start_time,
                failure_reason=(
                    None
                    if success
                    else "LIFT_TARGET_NOT_REACHED"
                ),
            )

        except Exception as e:
            return SkillResult(
                skill="LIFT_OBJECT",
                success=False,
                target=None,
                actual=None,
                error=None,
                duration=time.time() - start_time,
                failure_reason=(
                    "LIFT_EXECUTION_ERROR:"
                    + str(e)
                ),
            )

    def find_reachable_grasp_pose(
        self,
        contact_point,
        support_z,
        approach_distance,
        strategy="top",
        max_tilt_deg=45,
        step_deg=10,
        timeout=3.0
    ):
        """
        Find a reachable grasp pose while respecting the detected
        support surface and the physical finger envelope.
        """
        import math

        contact_point = np.asarray(
            contact_point,
            dtype=float
        )

        envelope = self.get_grasp_tool_envelope()

        if envelope is None:
            return {
                "success": False,
                "failure_reason": "GRASP_TOOL_ENVELOPE_UNAVAILABLE"
            }

        near_offset = float(
            envelope["near_offset"]
        )

        far_offset = float(
            envelope["far_offset"]
        )

        clearance = float(
            envelope["support_clearance"]
        )

        nominal_offset = 0.5 * (
            near_offset + far_offset
        )

        def quat_mul(q1, q2):
            x1, y1, z1, w1 = q1
            x2, y2, z2, w2 = q2

            return np.array([
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
                w1*w2 - x1*x2 - y1*y2 - z1*z2
            ], dtype=float)

        def quat_to_matrix(q):
            x, y, z, w = q

            return np.array([
                [
                    1 - 2*(y*y + z*z),
                    2*(x*y - z*w),
                    2*(x*z + y*w)
                ],
                [
                    2*(x*y + z*w),
                    1 - 2*(x*x + z*z),
                    2*(y*z - x*w)
                ],
                [
                    2*(x*z - y*w),
                    2*(y*z + x*w),
                    1 - 2*(x*x + y*y)
                ]
            ], dtype=float)

        base_q = np.array([
            1.0,
            0.0,
            0.0,
            0.0
        ], dtype=float)

        strategy_name = str(
            strategy
        ).strip().lower()

        nominal_strategies = {
            "top",
            "top_down",
        }

        alternate_strategies = {
            "adjust_grasp_angle_and_approach",
            "angled",
        }

        if strategy_name in nominal_strategies:
            candidates = [0]

            for angle in range(
                step_deg,
                max_tilt_deg + 1,
                step_deg
            ):
                candidates.extend([
                    angle,
                    -angle
                ])

        elif strategy_name in alternate_strategies:
            # Deliberately exclude the nominal 0-degree grasp.
            # The runtime searches a different reachable orientation
            # family while still deriving all poses internally.
            candidates = []

            for angle in range(
                step_deg,
                max_tilt_deg + 1,
                step_deg
            ):
                candidates.extend([
                    angle,
                    -angle
                ])

        else:
            return {
                "success": False,
                "failure_reason": (
                    "UNSUPPORTED_GRASP_STRATEGY:"
                    + strategy_name
                )
            }

        for deg in candidates:

            a = math.radians(deg) / 2.0

            tilt_q = np.array([
                0.0,
                math.sin(a),
                0.0,
                math.cos(a)
            ], dtype=float)

            q = quat_mul(
                base_q,
                tilt_q
            )

            q /= np.linalg.norm(q)

            R = quat_to_matrix(q)

            tool_axis = R @ np.array([
                0.0,
                0.0,
                1.0
            ])

            axis_down = -float(tool_axis[2])

            if axis_down <= 1e-4:
                continue

            minimum_tip_z = (
                float(support_z)
                + clearance
            )

            allowed_drop = (
                float(contact_point[2])
                - minimum_tip_z
            )

            required_contact_offset = (
                far_offset
                - allowed_drop / axis_down
            )

            contact_offset = max(
                nominal_offset,
                required_contact_offset
            )

            if (
                contact_offset < near_offset
                or contact_offset > far_offset
            ):
                continue

            grasp_position = (
                contact_point
                - tool_axis * contact_offset
            )

            fingertip_position = (
                grasp_position
                + tool_axis * far_offset
            )

            if (
                fingertip_position[2]
                < minimum_tip_z - 1e-6
            ):
                continue

            # Grasp pose itself must be feasible first.
            try:
                grasp_joints = self.solve_ik_pose(
                    grasp_position[0],
                    grasp_position[1],
                    grasp_position[2],
                    q,
                    timeout=timeout
                )
            except RuntimeError:
                continue

            # Preferred approach distance is a request, not a hard
            # requirement. Search for the largest reachable retreat.
            preferred_distance = float(
                approach_distance
            )

            approach_candidates = []

            d = preferred_distance

            while d >= 0.02 - 1e-9:
                approach_candidates.append(
                    round(d, 4)
                )
                d -= 0.01

            for actual_distance in approach_candidates:

                approach_position = (
                    grasp_position
                    - tool_axis * actual_distance
                )

                try:
                    approach_joints = self.solve_ik_pose(
                        approach_position[0],
                        approach_position[1],
                        approach_position[2],
                        q,
                        timeout=timeout
                    )
                except RuntimeError:
                    continue

                return {
                    "success": True,
                    "tilt_deg": float(deg),
                    "orientation": q,
                    "tool_axis": tool_axis,
                    "contact_offset": float(
                        contact_offset
                    ),
                    "requested_approach_distance": preferred_distance,
                    "actual_approach_distance": float(
                        actual_distance
                    ),
                    "grasp_target": grasp_position,
                    "approach_target": approach_position,
                    "fingertip_target": fingertip_position,
                    "grasp_joint_solution": grasp_joints,
                    "approach_joint_solution": approach_joints
                }

        return {
            "success": False,
            "failure_reason": "NO_REACHABLE_GRASP_POSE"
        }

    def find_reachable_grasp_orientation(
        self,
        x,
        y,
        z,
        max_tilt_deg=45,
        step_deg=10,
        timeout=3.0
    ):
        """
        Search orientations around the preferred top-down grasp pose.

        Returns the feasible orientation with the smallest angular
        deviation from top-down.
        """
        import math

        def quat_mul(q1, q2):
            x1, y1, z1, w1 = q1
            x2, y2, z2, w2 = q2

            return np.array([
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
                w1*w2 - x1*x2 - y1*y2 - z1*z2
            ], dtype=float)

        base_q = np.array([
            1.0,
            0.0,
            0.0,
            0.0
        ], dtype=float)

        candidates = [0]

        for angle in range(
            step_deg,
            max_tilt_deg + 1,
            step_deg
        ):
            candidates.extend([
                angle,
                -angle
            ])

        for deg in candidates:

            a = math.radians(deg) / 2.0

            tilt_q = np.array([
                0.0,
                math.sin(a),
                0.0,
                math.cos(a)
            ], dtype=float)

            q = quat_mul(
                base_q,
                tilt_q
            )

            q /= np.linalg.norm(q)

            try:
                joints = self.solve_ik_pose(
                    x,
                    y,
                    z,
                    q,
                    timeout=timeout
                )

                return {
                    "success": True,
                    "tilt_deg": float(deg),
                    "orientation": q,
                    "joint_solution": joints
                }

            except RuntimeError:
                continue

        return {
            "success": False,
            "failure_reason": "NO_REACHABLE_GRASP_ORIENTATION"
        }

    def solve_ik_pose(
        self,
        x,
        y,
        z,
        orientation,
        timeout=5.0
    ):
        self.wait_for_joint_state()

        if not self.ik_client.wait_for_service(
            timeout_sec=timeout
        ):
            raise RuntimeError(
                "IK_SERVICE_NOT_AVAILABLE"
            )

        request = GetPositionIK.Request()

        request.ik_request.group_name = self.move_group
        request.ik_request.ik_link_name = self.tool_frame
        request.ik_request.avoid_collisions = True

        request.ik_request.robot_state.joint_state.name = (
            self.joint_names
        )

        request.ik_request.robot_state.joint_state.position = (
            self.current_joint_position.tolist()
        )

        request.ik_request.pose_stamped.header.frame_id = (
            self.base_frame
        )

        request.ik_request.pose_stamped.pose.position.x = float(x)
        request.ik_request.pose_stamped.pose.position.y = float(y)
        request.ik_request.pose_stamped.pose.position.z = float(z)

        request.ik_request.pose_stamped.pose.orientation.x = float(
            orientation[0]
        )
        request.ik_request.pose_stamped.pose.orientation.y = float(
            orientation[1]
        )
        request.ik_request.pose_stamped.pose.orientation.z = float(
            orientation[2]
        )
        request.ik_request.pose_stamped.pose.orientation.w = float(
            orientation[3]
        )

        future = self.ik_client.call_async(request)

        rclpy.spin_until_future_complete(
            self,
            future
        )

        response = future.result()

        if response is None:
            raise RuntimeError(
                "IK_SERVICE_FAILED"
            )

        if response.error_code.val != 1:
            raise RuntimeError(
                f"IK_NOT_CONVERGED:{response.error_code.val}"
            )

        pos_map = dict(
            zip(
                response.solution.joint_state.name,
                response.solution.joint_state.position
            )
        )

        return np.array(
            [
                pos_map[name]
                for name in self.joint_names
            ],
            dtype=float
        )

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

        request = GetPositionIK.Request()

        request.ik_request.group_name = self.move_group
        request.ik_request.ik_link_name = self.tool_frame
        request.ik_request.avoid_collisions = True

        request.ik_request.robot_state.joint_state.name = (
            self.joint_names
        )

        request.ik_request.robot_state.joint_state.position = (
            self.current_joint_position.tolist()
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

        future = self.ik_client.call_async(
            request
        )

        rclpy.spin_until_future_complete(
            self,
            future
        )

        response = future.result()

        if response is None:
            raise RuntimeError(
                "IK_SERVICE_FAILED"
            )

        if response.error_code.val != 1:
            raise RuntimeError(
                f"IK_NOT_CONVERGED:{response.error_code.val}"
            )

        solution = response.solution.joint_state

        joint_map = dict(
            zip(
                solution.name,
                solution.position
            )
        )

        try:
            target_q = np.array([
                joint_map[name]
                for name in self.joint_names
            ])

        except KeyError as e:
            raise RuntimeError(
                f"IK_SOLUTION_MISSING_JOINT:{e}"
            )

        return target_q



    def plan_and_execute_joints(
        self,
        target_q,
        tolerance=0.01,
        timeout=10.0
    ):
        """
        Plan a joint-space motion through MoveIt before
        execution.

        This is the safety path for semantic joint motion.
        Direct move_joints() is not used here.
        """

        safety = self.verify_motion_safety_context()

        if not safety["success"]:
            return SkillResult(
                skill="MOVE_JOINTS_SAFE",
                success=False,
                target=np.asarray(
                    target_q,
                    dtype=float
                ),
                actual=np.array([]),
                error=0.0,
                duration=0.0,
                failure_reason=(
                    "SAFETY_PRECONDITION_FAILED:"
                    + str(
                        safety["failure_reason"]
                    )
                )
            )

        target_q = np.asarray(
            target_q,
            dtype=float
        )

        if target_q.shape != (len(self.joint_names),):
            return SkillResult(
                skill="MOVE_JOINTS_SAFE",
                success=False,
                target=target_q,
                actual=np.array([]),
                error=0.0,
                duration=0.0,
                failure_reason="INVALID_JOINT_TARGET"
            )

        self.wait_for_joint_state()

        start = time.time()

        request = GetMotionPlan.Request()

        motion_request = (
            request.motion_plan_request
        )

        motion_request.group_name = (
            self.move_group
        )

        motion_request.num_planning_attempts = 5
        motion_request.allowed_planning_time = float(
            timeout
        )

        motion_request.start_state.joint_state.name = (
            self.joint_names
        )

        motion_request.start_state.joint_state.position = (
            self.current_joint_position.tolist()
        )

        motion_request.start_state.is_diff = True

        constraints = Constraints()

        for joint_name, target in zip(
            self.joint_names,
            target_q
        ):
            joint_constraint = JointConstraint()

            joint_constraint.joint_name = (
                joint_name
            )

            joint_constraint.position = float(
                target
            )

            joint_constraint.tolerance_above = float(
                tolerance
            )

            joint_constraint.tolerance_below = float(
                tolerance
            )

            joint_constraint.weight = 1.0

            constraints.joint_constraints.append(
                joint_constraint
            )

        motion_request.goal_constraints = [
            constraints
        ]

        if not self.motion_plan_client.wait_for_service(
            timeout_sec=timeout
        ):
            return SkillResult(
                skill="MOVE_JOINTS_SAFE",
                success=False,
                target=target_q,
                actual=self.current_joint_position.copy(),
                error=0.0,
                duration=time.time() - start,
                failure_reason="MOTION_PLAN_SERVICE_UNAVAILABLE"
            )

        future = self.motion_plan_client.call_async(
            request
        )

        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=timeout
        )

        if not future.done():
            return SkillResult(
                skill="MOVE_JOINTS_SAFE",
                success=False,
                target=target_q,
                actual=self.current_joint_position.copy(),
                error=0.0,
                duration=time.time() - start,
                failure_reason="MOTION_PLAN_TIMEOUT"
            )

        response = future.result()

        if response is None:
            return SkillResult(
                skill="MOVE_JOINTS_SAFE",
                success=False,
                target=target_q,
                actual=self.current_joint_position.copy(),
                error=0.0,
                duration=time.time() - start,
                failure_reason="MOTION_PLAN_FAILED"
            )

        plan_response = (
            response.motion_plan_response
        )

        if plan_response.error_code.val != 1:
            return SkillResult(
                skill="MOVE_JOINTS_SAFE",
                success=False,
                target=target_q,
                actual=self.current_joint_position.copy(),
                error=0.0,
                duration=time.time() - start,
                failure_reason=(
                    "PLANNING_FAILED:"
                    + str(
                        plan_response.error_code.val
                    )
                )
            )

        trajectory = (
            plan_response.trajectory
        )

        if not trajectory.joint_trajectory.points:
            return SkillResult(
                skill="MOVE_JOINTS_SAFE",
                success=False,
                target=target_q,
                actual=self.current_joint_position.copy(),
                error=0.0,
                duration=time.time() - start,
                failure_reason="EMPTY_PLANNED_TRAJECTORY"
            )

        execute_result = (
            self.execute_planned_trajectory(
                trajectory
            )
        )

        self.wait_for_joint_state()

        actual = (
            self.current_joint_position.copy()
        )

        error = float(
            np.max(
                np.abs(
                    target_q - actual
                )
            )
        )

        success = (
            execute_result.success
            and error <= tolerance
        )

        return SkillResult(
            skill="MOVE_JOINTS_SAFE",
            success=success,
            target=target_q,
            actual=actual,
            error=error,
            duration=time.time() - start,
            failure_reason=(
                None
                if success
                else (
                    execute_result.failure_reason
                    if not execute_result.success
                    else "JOINT_TARGET_NOT_REACHED"
                )
            )
        )


    def execute_planned_trajectory(
        self,
        robot_trajectory,
        settle_timeout=6.0,
        rate_hz=100.0,
        time_scale=2.5
    ):

        traj = robot_trajectory.joint_trajectory

        if not traj.points:
            return SkillResult(
                skill="EXECUTE_TRAJECTORY",
                success=False,
                target=np.array([]),
                actual=np.array([]),
                error=0.0,
                duration=0.0,
                failure_reason="EMPTY_TRAJECTORY"
            )

        self.wait_for_joint_state()

        index = {
            name: i
            for i, name in enumerate(traj.joint_names)
        }

        try:
            ordered_points = [
                np.array([
                    point.positions[index[name]]
                    for name in self.joint_names
                ], dtype=float)
                for point in traj.points
            ]
        except KeyError:
            return SkillResult(
                skill="EXECUTE_TRAJECTORY",
                success=False,
                target=np.array([]),
                actual=np.array([]),
                error=0.0,
                duration=0.0,
                failure_reason="TRAJECTORY_JOINT_MISMATCH"
            )

        def stamp_sec(point):
            return (
                float(point.time_from_start.sec)
                + float(point.time_from_start.nanosec) / 1e9
            )

        times = [
            stamp_sec(point)
            for point in traj.points
        ]

        start = time.time()

        # Follow exactly the joint-space path generated by MoveIt.
        # Interpolate only between consecutive planned waypoints.
        for i in range(len(ordered_points) - 1):

            q0 = ordered_points[i]
            q1 = ordered_points[i + 1]

            t0 = times[i]
            t1 = times[i + 1]

            segment_duration = max(
                (t1 - t0) * float(time_scale),
                0.001
            )

            segment_start = time.time()

            while rclpy.ok():

                elapsed = (
                    time.time()
                    - segment_start
                )

                alpha = min(
                    elapsed / segment_duration,
                    1.0
                )

                q = (
                    q0
                    + alpha * (q1 - q0)
                )

                msg = Float64MultiArray()
                msg.data = q.tolist()

                self.position_pub.publish(msg)

                rclpy.spin_once(
                    self,
                    timeout_sec=1.0 / rate_hz
                )

                if alpha >= 1.0:
                    break

        target_q = ordered_points[-1]

        final_msg = Float64MultiArray()
        final_msg.data = target_q.tolist()

        settle_start = time.time()

        actual = self.current_joint_position.copy()
        error = float("inf")

        settle_tolerance = 0.003
        required_stable_time = 0.75
        stable_since = None

        while rclpy.ok():

            self.position_pub.publish(
                final_msg
            )

            rclpy.spin_once(
                self,
                timeout_sec=0.02
            )

            actual = (
                self.current_joint_position.copy()
            )

            error = float(
                np.max(
                    np.abs(
                        target_q - actual
                    )
                )
            )

            if error <= settle_tolerance:

                if stable_since is None:
                    stable_since = time.time()

                if (
                    time.time()
                    - stable_since
                    >= required_stable_time
                ):
                    break

            else:
                stable_since = None

            if (
                time.time()
                - settle_start
                > settle_timeout
            ):
                break

        return SkillResult(
            skill="EXECUTE_TRAJECTORY",
            success=(error <= settle_tolerance),
            target=target_q,
            actual=actual,
            error=error,
            duration=time.time() - start,
            failure_reason=(
                None
                if error <= settle_tolerance
                else "TRAJECTORY_TARGET_NOT_REACHED"
            )
        )

    def plan_tcp_pose(
        self,
        target,
        orientation,
        position_tolerance=0.008,
        orientation_tolerance=0.10,
        timeout=5.0
    ):
        """
        Collision-aware TCP pose planning only.

        Returns a MoveIt RobotTrajectory without executing it.
        All collision validation is performed by MoveIt against
        the current PlanningScene.
        """

        safety = self.verify_motion_safety_context()

        if not safety["success"]:
            return {
                "success": False,
                "failure_reason": (
                    "SAFETY_PRECONDITION_FAILED:"
                    + str(
                        safety["failure_reason"]
                    )
                ),
                "safety_context": safety,
            }

        target = np.asarray(
            target,
            dtype=float
        )

        orientation = np.asarray(
            orientation,
            dtype=float
        )

        if target.shape != (3,):
            return {
                "success": False,
                "failure_reason":
                    "INVALID_TCP_TARGET"
            }

        if orientation.shape != (4,):
            return {
                "success": False,
                "failure_reason":
                    "INVALID_TCP_ORIENTATION"
            }

        quat_norm = float(
            np.linalg.norm(
                orientation
            )
        )

        if quat_norm <= 1e-9:
            return {
                "success": False,
                "failure_reason":
                    "INVALID_TCP_ORIENTATION"
            }

        orientation = (
            orientation / quat_norm
        )

        self.wait_for_joint_state()

        request = GetMotionPlan.Request()

        req = (
            request.motion_plan_request
        )

        req.group_name = self.move_group

        req.num_planning_attempts = 5

        req.allowed_planning_time = float(
            timeout
        )

        #
        # Explicit current start state.
        #
        req.start_state.joint_state.name = (
            self.joint_names
        )

        req.start_state.joint_state.position = (
            self.current_joint_position.tolist()
        )

        req.start_state.is_diff = True

        constraints = Constraints()
        constraints.name = "safe_tcp_pose_goal"

        #
        # Position constraint.
        #
        pos = PositionConstraint()

        pos.header.frame_id = (
            self.base_frame
        )

        pos.link_name = (
            self.tool_frame
        )

        pos.weight = 1.0

        region = BoundingVolume()

        primitive = SolidPrimitive()

        primitive.type = (
            SolidPrimitive.SPHERE
        )

        primitive.dimensions = [
            float(position_tolerance)
        ]

        pose = Pose()

        pose.position.x = float(
            target[0]
        )

        pose.position.y = float(
            target[1]
        )

        pose.position.z = float(
            target[2]
        )

        pose.orientation.w = 1.0

        region.primitives = [
            primitive
        ]

        region.primitive_poses = [
            pose
        ]

        pos.constraint_region = region

        #
        # Orientation constraint.
        #
        ori = OrientationConstraint()

        ori.header.frame_id = (
            self.base_frame
        )

        ori.link_name = (
            self.tool_frame
        )

        ori.orientation.x = float(
            orientation[0]
        )

        ori.orientation.y = float(
            orientation[1]
        )

        ori.orientation.z = float(
            orientation[2]
        )

        ori.orientation.w = float(
            orientation[3]
        )

        ori.absolute_x_axis_tolerance = float(
            orientation_tolerance
        )

        ori.absolute_y_axis_tolerance = float(
            orientation_tolerance
        )

        ori.absolute_z_axis_tolerance = float(
            orientation_tolerance
        )

        ori.weight = 1.0

        constraints.position_constraints = [
            pos
        ]

        constraints.orientation_constraints = [
            ori
        ]

        req.goal_constraints = [
            constraints
        ]

        if not self.motion_plan_client.wait_for_service(
            timeout_sec=timeout
        ):
            return {
                "success": False,
                "failure_reason":
                    "MOTION_PLAN_SERVICE_UNAVAILABLE"
            }

        future = (
            self.motion_plan_client.call_async(
                request
            )
        )

        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=timeout + 2.0
        )

        if not future.done():
            return {
                "success": False,
                "failure_reason":
                    "MOTION_PLAN_TIMEOUT"
            }

        response = future.result()

        if response is None:
            return {
                "success": False,
                "failure_reason":
                    "NO_PLANNING_RESPONSE"
            }

        plan = (
            response.motion_plan_response
        )

        if plan.error_code.val != 1:
            return {
                "success": False,
                "failure_reason": (
                    "PLANNING_FAILED:"
                    + str(
                        plan.error_code.val
                    )
                )
            }

        trajectory = plan.trajectory

        if (
            not trajectory
            .joint_trajectory
            .points
        ):
            return {
                "success": False,
                "failure_reason":
                    "EMPTY_PLANNED_TRAJECTORY"
            }

        return {
            "success": True,
            "trajectory": trajectory,
            "target": target,
            "orientation": orientation,
            "planned_at_monotonic": time.monotonic(),
        }


    def plan_and_execute_tcp(
        self,
        x,
        y,
        z,
        tolerance=0.01,
        timeout=10.0
    ):

        from moveit_msgs.msg import (
            Constraints,
            PositionConstraint,
            OrientationConstraint,
            BoundingVolume
        )
        from shape_msgs.msg import SolidPrimitive
        from geometry_msgs.msg import Pose

        start = time.time()

        current_xyz, current_quat = self.get_tcp_pose()

        request = GetMotionPlan.Request()

        motion_request = request.motion_plan_request
        motion_request.group_name = self.move_group
        motion_request.num_planning_attempts = 5
        motion_request.allowed_planning_time = 5.0

        constraints = Constraints()
        constraints.name = "tcp_goal"

        pos = PositionConstraint()
        pos.header.frame_id = self.base_frame
        pos.link_name = self.tool_frame
        pos.weight = 1.0

        region = BoundingVolume()

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [
            float(tolerance)
        ]

        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        pose.orientation.w = 1.0

        region.primitives = [primitive]
        region.primitive_poses = [pose]

        pos.constraint_region = region

        ori = OrientationConstraint()
        ori.header.frame_id = self.base_frame
        ori.link_name = self.tool_frame

        ori.orientation.x = float(current_quat[0])
        ori.orientation.y = float(current_quat[1])
        ori.orientation.z = float(current_quat[2])
        ori.orientation.w = float(current_quat[3])

        ori.absolute_x_axis_tolerance = 0.15
        ori.absolute_y_axis_tolerance = 0.15
        ori.absolute_z_axis_tolerance = 0.15
        ori.weight = 1.0

        constraints.position_constraints = [pos]
        constraints.orientation_constraints = [ori]

        motion_request.goal_constraints = [
            constraints
        ]

        if not self.motion_plan_client.wait_for_service(
            timeout_sec=timeout
        ):
            return SkillResult(
                skill="MOVE_TCP",
                success=False,
                target=np.array([x, y, z]),
                actual=np.array([]),
                error=0.0,
                duration=time.time() - start,
                failure_reason="MOTION_PLAN_SERVICE_UNAVAILABLE"
            )

        future = self.motion_plan_client.call_async(
            request
        )

        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=timeout
        )

        if not future.done():
            return SkillResult(
                skill="MOVE_TCP",
                success=False,
                target=np.array([x, y, z]),
                actual=np.array([]),
                error=0.0,
                duration=time.time() - start,
                failure_reason="MOTION_PLAN_TIMEOUT"
            )

        response = future.result()

        if response is None:
            return SkillResult(
                skill="MOVE_TCP",
                success=False,
                target=np.array([x, y, z]),
                actual=np.array([]),
                error=0.0,
                duration=time.time() - start,
                failure_reason="MOTION_PLAN_FAILED"
            )

        plan_response = response.motion_plan_response

        if plan_response.error_code.val != 1:
            return SkillResult(
                skill="MOVE_TCP",
                success=False,
                target=np.array([x, y, z]),
                actual=np.array([]),
                error=0.0,
                duration=time.time() - start,
                failure_reason=(
                    "PLANNING_FAILED:"
                    + str(plan_response.error_code.val)
                )
            )

        execute_result = (
            self.execute_planned_trajectory(
                plan_response.trajectory
            )
        )

        if not execute_result.success:
            return SkillResult(
                skill="MOVE_TCP",
                success=False,
                target=np.array([x, y, z]),
                actual=np.array([]),
                error=0.0,
                duration=time.time() - start,
                failure_reason=(
                    execute_result.failure_reason
                )
            )

        actual_xyz, _ = self.get_tcp_pose()

        target_xyz = np.array(
            [x, y, z],
            dtype=float
        )

        error = float(
            np.linalg.norm(
                target_xyz - actual_xyz
            )
        )

        success = error <= tolerance

        return SkillResult(
            skill="MOVE_TCP",
            success=success,
            target=target_xyz,
            actual=actual_xyz,
            error=error,
            duration=time.time() - start,
            failure_reason=(
                None
                if success
                else "TCP_TARGET_NOT_REACHED"
            )
        )

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

        try:
            _, current_orientation = self.get_tcp_pose(
                timeout=timeout,
            )
        except Exception as exc:
            return SkillResult(
                skill="MOVE_TCP",
                success=False,
                target=target_xyz,
                actual=np.array([]),
                error=0.0,
                duration=time.time() - start,
                failure_reason="TCP_POSE_UNAVAILABLE:" + str(exc),
            )

        plan = self.plan_tcp_pose(
            target_xyz,
            current_orientation,
            position_tolerance=tolerance,
            timeout=timeout,
        )

        if not plan["success"]:
            return SkillResult(
                skill="MOVE_TCP",
                success=False,
                target=target_xyz,
                actual=np.array([]),
                error=0.0,
                duration=time.time() - start,
                failure_reason=plan["failure_reason"],
            )

        gate = ExecutionAuthorizationGate(self)
        authorization = gate.authorize(
            plan["trajectory"],
            planned_at_monotonic=plan["planned_at_monotonic"],
        )

        if not authorization.authorized:
            return SkillResult(
                skill="MOVE_TCP",
                success=False,
                target=target_xyz,
                actual=np.array([]),
                error=0.0,
                duration=time.time() - start,
                failure_reason=(
                    "EXECUTION_AUTHORIZATION_DENIED:"
                    + authorization.reason
                ),
            )

        authorized, reason = gate.verify_authorization(
            plan["trajectory"],
            authorization,
        )

        if not authorized:
            return SkillResult(
                skill="MOVE_TCP",
                success=False,
                target=target_xyz,
                actual=np.array([]),
                error=0.0,
                duration=time.time() - start,
                failure_reason=(
                    "EXECUTION_AUTHORIZATION_INVALID:"
                    + reason
                ),
            )

        trajectory_result = self.execute_planned_trajectory(
            plan["trajectory"],
        )

        if not trajectory_result.success:
            return SkillResult(
                skill="MOVE_TCP",
                success=False,
                target=target_xyz,
                actual=np.array([]),
                error=0.0,
                duration=time.time() - start,
                failure_reason=trajectory_result.failure_reason,
            )

        actual_xyz, _ = self.get_tcp_pose()

        error = float(
            np.linalg.norm(
                target_xyz - actual_xyz
            )
        )

        success = (
            error <= tolerance
        )

        return SkillResult(
            skill="MOVE_TCP",
            success=success,
            target=target_xyz,
            actual=actual_xyz,
            error=error,
            duration=time.time() - start,
            failure_reason=(
                None
                if success
                else "TARGET_NOT_REACHED"
            )
        )
