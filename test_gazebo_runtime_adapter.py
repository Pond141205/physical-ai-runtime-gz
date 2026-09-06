import rclpy
import numpy as np

from gazebo_runtime_adapter import GazeboUR5eAdapter


def main():
    rclpy.init()

    robot = GazeboUR5eAdapter()

    print("Capabilities:")
    print(robot.get_capabilities())

    state = robot.get_state()

    print("\nCurrent joints:")
    print(state.joint_position)

    target = np.array([
        0.0,
        -1.0,
        1.2,
        -1.4,
        -1.57,
        0.0
    ])

    print("\nSending target...")

    result = robot.move_joints(
        target,
        duration=3.0
    )

    print("\nResult:")
    print("success :", result.success)
    print("error   :", result.error)
    print("actual  :", result.actual)
    print("failure :", result.failure_reason)

    robot.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
