import rclpy
import numpy as np

from runtime import RobotRuntime
from skills import MoveTCP
from gazebo_runtime_adapter import GazeboUR5eAdapter


def main():

    rclpy.init()

    robot = GazeboUR5eAdapter()
    runtime = RobotRuntime()

    skill = MoveTCP(
        position=np.array([
            0.60,
            0.05,
            0.35
        ]),
        duration=3.0,
        tolerance=0.01
    )

    print("Executing semantic MoveTCP...")

    result = runtime.execute(
        robot,
        skill
    )

    print("\n--- RUNTIME RESULT ---")

    print("success :", result.success)
    print("target  :", result.target)
    print("actual  :", result.actual)
    print("error   :", result.error)
    print("duration:", result.duration)
    print("failure :", result.failure_reason)

    robot.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()