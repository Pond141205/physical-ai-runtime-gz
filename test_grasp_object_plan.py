import rclpy

from runtime import RobotRuntime
from gazebo_panda_adapter import GazeboPandaAdapter
from skills import GraspObject


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
