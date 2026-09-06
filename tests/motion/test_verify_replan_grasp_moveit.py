import numpy as np
import rclpy

from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.msg import (
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
)
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose

from physical_ai_runtime.adapters.gazebo_panda_adapter import GazeboPandaAdapter


TARGET = np.array([
    0.23151818,
    -0.35974636,
    0.83754412
])

TARGET_QUAT = np.array([
    0.99619470,
    0.0,
    0.08715574,
    0.0
])


rclpy.init()
robot = GazeboPandaAdapter()

try:
    current_xyz, _ = robot.get_tcp_pose()

    request = GetMotionPlan.Request()
    req = request.motion_plan_request

    req.group_name = robot.move_group
    req.num_planning_attempts = 5
    req.allowed_planning_time = 5.0

    constraints = Constraints()
    constraints.name = "verify_replanned_grasp"

    pos = PositionConstraint()
    pos.header.frame_id = robot.base_frame
    pos.link_name = robot.tool_frame
    pos.weight = 1.0

    region = BoundingVolume()

    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.SPHERE
    primitive.dimensions = [0.008]

    pose = Pose()
    pose.position.x = float(TARGET[0])
    pose.position.y = float(TARGET[1])
    pose.position.z = float(TARGET[2])
    pose.orientation.w = 1.0

    region.primitives = [primitive]
    region.primitive_poses = [pose]

    pos.constraint_region = region

    ori = OrientationConstraint()
    ori.header.frame_id = robot.base_frame
    ori.link_name = robot.tool_frame

    ori.orientation.x = float(TARGET_QUAT[0])
    ori.orientation.y = float(TARGET_QUAT[1])
    ori.orientation.z = float(TARGET_QUAT[2])
    ori.orientation.w = float(TARGET_QUAT[3])

    ori.absolute_x_axis_tolerance = 0.10
    ori.absolute_y_axis_tolerance = 0.10
    ori.absolute_z_axis_tolerance = 0.10
    ori.weight = 1.0

    constraints.position_constraints = [pos]
    constraints.orientation_constraints = [ori]

    req.goal_constraints = [constraints]

    print("CURRENT TCP:")
    print(current_xyz)

    print("\nGRASP TARGET:")
    print(TARGET)

    print("\nTARGET QUAT:")
    print(TARGET_QUAT)

    if not robot.motion_plan_client.wait_for_service(
        timeout_sec=5.0
    ):
        raise RuntimeError(
            "MOTION_PLAN_SERVICE_UNAVAILABLE"
        )

    future = robot.motion_plan_client.call_async(
        request
    )

    rclpy.spin_until_future_complete(
        robot,
        future,
        timeout_sec=10.0
    )

    if not future.done():
        raise RuntimeError(
            "MOTION_PLAN_TIMEOUT"
        )

    response = future.result()

    if response is None:
        raise RuntimeError(
            "NO_PLANNING_RESPONSE"
        )

    plan = response.motion_plan_response

    print("\nMOVEIT ERROR CODE:")
    print(plan.error_code.val)

    print("\nTRAJECTORY POINTS:")
    print(
        len(
            plan.trajectory
            .joint_trajectory
            .points
        )
    )

    if plan.trajectory.joint_trajectory.points:
        print("\nLAST POINT:")
        print(
            plan.trajectory
            .joint_trajectory
            .points[-1]
            .positions
        )

finally:
    robot.destroy_node()
    rclpy.shutdown()
