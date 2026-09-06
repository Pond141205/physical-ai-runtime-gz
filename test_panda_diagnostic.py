import time
import numpy as np
import rclpy

from gazebo_panda_adapter import GazeboPandaAdapter


def spin_for(node, seconds):
    end = time.time() + seconds
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.05)


def main():
    rclpy.init()
    panda = GazeboPandaAdapter()

    try:
        print("Waiting for Panda state...")
        spin_for(panda, 2.0)

        print("\nTCP before:")
        print(panda.get_tcp_pose())

        print("\nJoint state before:")
        print(panda.get_state().joint_position)

        target = np.array([0.47, 0.02, 0.47])

        print("\nSolving IK for:")
        print(target)

        ik_joints = panda.solve_ik(
    target[0],
    target[1],
    target[2]
)

        print("\nIK joint target:")
        print(ik_joints)

        print("\nExecuting joint target...")

        result = panda.move_joints(
            ik_joints,
            duration=3.0
        )

        print("\nJoint result:")
        print("success :", result.success)
        print("target  :", result.target)
        print("actual  :", result.actual)
        print("error   :", result.error)
        print("failure :", result.failure_reason)

        spin_for(panda, 0.5)

        print("\nTCP after:")
        tcp = panda.get_tcp_pose()
        print(tcp)

        actual_position = np.array(tcp[0])
        error = np.linalg.norm(actual_position - target)

        print("\nTCP target :", target)
        print("TCP actual :", actual_position)
        print("TCP error  :", error)

    finally:
        panda.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
