import rclpy

from physical_ai_runtime.core.runtime import RobotRuntime
from physical_ai_runtime.adapters.gazebo_panda_adapter import GazeboPandaAdapter
from physical_ai_runtime.core.skills import GraspObject


rclpy.init()

robot = GazeboPandaAdapter()
runtime = RobotRuntime()

skill = GraspObject(
    object_id="cube",
    strategy="top",
    approach_height=0.08,
    lift_height=0.08
)

result = runtime.plan_grasp_object(
    robot,
    skill
)

print()
print("GRASP OBJECT PLAN:")
print(result)

robot.destroy_node()
rclpy.shutdown()
