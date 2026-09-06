import rclpy

from gazebo_runtime_adapter import GazeboUR5eAdapter


def main():

    rclpy.init()

    robot = GazeboUR5eAdapter()

    position, orientation = robot.get_tcp_pose()

    print("Current TCP:")
    print(position)

    result = robot.execute_move(
        x=0.55,
        y=0.10,
        z=0.40,
        duration=3.0
    )

    print("\n--- RESULT ---")
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