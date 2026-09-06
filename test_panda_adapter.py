import rclpy
import numpy as np

from gazebo_panda_adapter import (
    GazeboPandaAdapter
)


def main():

    rclpy.init()

    robot = GazeboPandaAdapter()

    print("Capabilities:")
    print(robot.get_capabilities())

    print("\nCurrent joints:")

    state = robot.get_state()

    print(state.joint_position)

    target = np.array([
        0.0,
        -0.6,
        0.0,
        -2.0,
        0.0,
        1.5,
        0.7
    ])

    print("\nMoving Panda...")

    result = robot.move_joints(
        target,
        duration=3.0
    )

    print("\n--- RESULT ---")
    print("success :", result.success)
    print("error   :", result.error)
    print("actual  :", result.actual)
    print("failure :", result.failure_reason)

    robot.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()