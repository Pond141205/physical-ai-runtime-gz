import rclpy
import numpy as np

from physical_ai_runtime.core.runtime import RobotRuntime
from physical_ai_runtime.adapters.gazebo_panda_adapter import GazeboPandaAdapter
from physical_ai_runtime.core.skills import GraspObject
from physical_ai_runtime.perception.scene_observer import SceneObserver


rclpy.init()

robot = GazeboPandaAdapter()
runtime = RobotRuntime()

print("\n[0] GO HOME")

home_result = robot.go_home(
    duration=4.0
)

print("HOME RESULT:")
print(home_result)

home_tcp, _ = robot.get_tcp_pose()

print("HOME TCP:")
print(home_tcp)

skill = GraspObject(
    object_id="cube",
    strategy="top",
    approach_height=0.08,
    lift_height=0.08
)

print("\n[1] OBSERVE + PLAN")

plan = runtime.plan_grasp_object(
    robot,
    skill
)

print(plan)

if not plan["success"]:
    print("PLAN FAILED")
else:
    target = np.array(
        plan["approach_target"],
        dtype=float
    )

    print("\n[2] MOVE TO PERCEPTION-DERIVED APPROACH")
    print("TARGET:", target)

    result = robot.execute_move(
        *target,
        tolerance=0.01,
        timeout=5.0,
        duration=3.0
    )

    print("MOVE RESULT:")
    print(result)

    print("\n[3] RE-OBSERVE")

    observer = SceneObserver()

    try:
        scene = observer.observe_once(
            timeout=5.0
        )

        cube = scene.objects["cube"]

        print("CUBE AFTER APPROACH:")
        print(cube.position_robot)

    finally:
        observer.destroy_node()

robot.destroy_node()
rclpy.shutdown()
