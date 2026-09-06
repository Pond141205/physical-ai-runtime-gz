import time
import numpy as np
import rclpy

from physical_ai_runtime.adapters.gazebo_runtime_adapter import GazeboUR5eAdapter


def spin_for(node, seconds):
    end = time.time() + seconds
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.05)


def main():
    rclpy.init()
    ur = GazeboUR5eAdapter()

    try:
        print("Waiting for UR5e state...")
        spin_for(ur, 2.0)

        print("\nTCP before:")
        print(ur.get_tcp_pose())

        print("\nJoint state before:")
        print(ur.get_state().joint_position)

        target = np.array([0.62, 0.07, 0.37])

        print("\nSolving IK for:")
        print(target)

        ik_joints = ur.solve_ik(
            target[0],
            target[1],
            target[2]
        )

        print("\nIK joint target:")
        print(ik_joints)

        print("\nExecuting joint target...")

        result = ur.move_joints(
            ik_joints,
            duration=3.0
        )

        print("\nJoint result:")
        print("success :", result.success)
        print("target  :", result.target)
        print("actual  :", result.actual)
        print("error   :", result.error)
        print("failure :", result.failure_reason)

        spin_for(ur, 0.5)

        print("\nJoint actual after:")
        print(ur.get_state().joint_position)

        print("\nTCP after:")
        tcp = ur.get_tcp_pose()
        print(tcp)

        actual = np.array(tcp[0])

        print("\nTCP target :", target)
        print("TCP actual :", actual)
        print("TCP error  :", np.linalg.norm(actual - target))

    finally:
        ur.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
