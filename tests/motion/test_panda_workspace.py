import rclpy
import numpy as np

from physical_ai_runtime.core.runtime import RobotRuntime
from physical_ai_runtime.core.skills import MoveTCP
from physical_ai_runtime.adapters.gazebo_panda_adapter import GazeboPandaAdapter


def main():

    rclpy.init()

    robot = GazeboPandaAdapter()
    runtime = RobotRuntime()

    skill = MoveTCP(
        position=np.array([
            0.02,
            0.02,
            0.02
        ]),
        frame="workspace",
        duration=3.0,
        tolerance=0.01
    )

    print("Executing Panda workspace MoveTCP...")

    result = runtime.execute(
        robot,
        skill
    )

    print("\n--- RESULT ---")
    print("success :", result.success)
    print("target  :", result.target)
    print("actual  :", result.actual)
    print("error   :", result.error)
    print("failure :", result.failure_reason)

    robot.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()