import rclpy
import numpy as np

from physical_ai_runtime.adapters.gazebo_ur5e_adapter import GazeboUR5eAdapter


def main():
    rclpy.init()

    robot = GazeboUR5eAdapter()

    target = np.array([
        0.0,
        -1.2,
        1.4,
        -1.5,
        -1.57,
        0.0
    ])

    print("Sending joint trajectory...")

    result = robot.move_joints(
        target,
        duration=3.0
    )

    print("Done")
    print(result)

    robot.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
