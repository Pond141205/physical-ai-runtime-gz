import rclpy

from rclpy.node import Node

from moveit_msgs.srv import ApplyPlanningScene
from moveit_msgs.msg import PlanningScene, CollisionObject

from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose

from workspace_observer import WorkspaceObserver


SAFETY_MARGIN = 0.04
TABLE_THICKNESS = 0.05


def main():
    rclpy.init()

    observer = WorkspaceObserver()

    workspace = observer.observe_once(
        timeout=5.0
    )

    observer.destroy_node()

    if workspace is None:
        raise RuntimeError(
            "WORKSPACE_NOT_DETECTED"
        )

    print("DETECTED WORKSPACE:")
    print(workspace)

    x_min = workspace["x_min"]
    x_max = workspace["x_max"]
    y_min = workspace["y_min"]
    y_max = workspace["y_max"]
    table_z = workspace["table_z"]

    width = x_max - x_min
    length = y_max - y_min

    center_x = 0.5 * (
        x_min + x_max
    )

    center_y = 0.5 * (
        y_min + y_max
    )

    safe_workspace = {
        "x_min": x_min + SAFETY_MARGIN,
        "x_max": x_max - SAFETY_MARGIN,
        "y_min": y_min + SAFETY_MARGIN,
        "y_max": y_max - SAFETY_MARGIN,
        "z": table_z,
    }

    print()
    print("SAFE WORKSPACE:")
    print(safe_workspace)

    node = Node(
        "workspace_scene_sync"
    )

    client = node.create_client(
        ApplyPlanningScene,
        "/panda/apply_planning_scene"
    )

    if not client.wait_for_service(
        timeout_sec=5.0
    ):
        raise RuntimeError(
            "APPLY_PLANNING_SCENE_NOT_AVAILABLE"
        )

    collision = CollisionObject()

    collision.header.frame_id = "world"
    collision.id = "detected_support_surface"

    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    # Conservative collision geometry:
    # expand detected table boundary outward so planning
    # keeps robot links away from uncertain physical edges.
    collision_width = width + 2.0 * SAFETY_MARGIN
    collision_length = length + 2.0 * SAFETY_MARGIN

    box.dimensions = [
        float(collision_width),
        float(collision_length),
        float(TABLE_THICKNESS)
    ]

    pose = Pose()

    pose.position.x = float(center_x)
    pose.position.y = float(center_y)

    # table_z is detected top surface
    pose.position.z = float(
        table_z
        - TABLE_THICKNESS / 2.0
    )

    pose.orientation.w = 1.0

    collision.primitives = [box]
    collision.primitive_poses = [pose]

    collision.operation = (
        CollisionObject.ADD
    )

    scene = PlanningScene()
    scene.is_diff = True

    scene.world.collision_objects = [
        collision
    ]

    req = ApplyPlanningScene.Request()
    req.scene = scene

    future = client.call_async(req)

    rclpy.spin_until_future_complete(
        node,
        future,
        timeout_sec=5.0
    )

    response = future.result()

    if response is None:
        raise RuntimeError(
            "NO_PLANNING_SCENE_RESPONSE"
        )

    print()
    print(
        "PLANNING SCENE APPLIED:",
        response.success
    )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
