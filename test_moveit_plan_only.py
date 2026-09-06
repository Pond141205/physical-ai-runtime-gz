import rclpy
import numpy as np

from gazebo_panda_adapter import GazeboPandaAdapter
from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.msg import (
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
)
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose


TARGET = np.array([
    0.24798026,
    -0.35110616,
    0.90539037
])


rclpy.init()
robot = GazeboPandaAdapter()

try:
    current_xyz, current_quat = robot.get_tcp_pose()

    request = GetMotionPlan.Request()
    req = request.motion_plan_request

    req.group_name = robot.move_group
    req.num_planning_attempts = 5
    req.allowed_planning_time = 5.0

    constraints = Constraints()
    constraints.name = "tcp_goal"

    pos = PositionConstraint()
    pos.header.frame_id = robot.base_frame
    pos.link_name = robot.tool_frame
    pos.weight = 1.0

    region = BoundingVolume()

    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.SPHERE
    primitive.dimensions = [0.01]

    pose = Pose()
    pose.position.x = float(TARGET[0])
    pose.position.y = float(TARGET[1])
    pose.position.z = float(TARGET[2])
    pose.orientation.w = 1.0

    region.primitives = [primitive]
    region.primitive_poses = [pose]

    pos.constraint_region = region

    orientation_result = robot.find_reachable_grasp_orientation(
        float(TARGET[0]),
        float(TARGET[1]),
        float(TARGET[2]),
    )

    if not orientation_result["success"]:
        raise RuntimeError(
            orientation_result["failure_reason"]
        )

    target_quat = orientation_result[
        "orientation"
    ]

    print(
        "\nSELECTED TILT:",
        orientation_result["tilt_deg"]
    )

    ori = OrientationConstraint()
    ori.header.frame_id = robot.base_frame
    ori.link_name = robot.tool_frame
    ori.orientation.x = float(target_quat[0])
    ori.orientation.y = float(target_quat[1])
    ori.orientation.z = float(target_quat[2])
    ori.orientation.w = float(target_quat[3])

    ori.absolute_x_axis_tolerance = 0.15
    ori.absolute_y_axis_tolerance = 0.15
    ori.absolute_z_axis_tolerance = 0.15
    ori.weight = 1.0

    constraints.position_constraints = [pos]
    constraints.orientation_constraints = [ori]

    req.goal_constraints = [constraints]

    print("CURRENT TCP:")
    print(current_xyz)

    print("\nTARGET TCP:")
    print(TARGET)

    print("\nTARGET QUAT:")
    print(target_quat)

    print("\nWAITING FOR PLANNER...")

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

    trajectory = plan.trajectory.joint_trajectory

    print("\nTRAJECTORY JOINTS:")
    print(trajectory.joint_names)

    print("\nTRAJECTORY POINTS:")
    print(len(trajectory.points))

    if trajectory.points:
        print("\nFIRST POINT:")
        print(trajectory.points[0].positions)

        print("\nLAST POINT:")
        print(trajectory.points[-1].positions)

finally:
    robot.destroy_node()
    rclpy.shutdown()
