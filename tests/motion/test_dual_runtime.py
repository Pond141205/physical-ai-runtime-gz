import time
import numpy as np
import rclpy

from physical_ai_runtime.core.runtime import RobotRuntime
from physical_ai_runtime.core.skills import MoveTCP
from physical_ai_runtime.adapters.gazebo_runtime_adapter import GazeboUR5eAdapter
from physical_ai_runtime.adapters.gazebo_panda_adapter import GazeboPandaAdapter


def wait_for_state(adapter, name, timeout=10.0):
    start = time.time()

    while time.time() - start < timeout:
        rclpy.spin_once(adapter, timeout_sec=0.1)

        try:
            state = adapter.get_state()
        except Exception:
            state = None

        if state is not None:
            print(f"{name} state ready")
            return True

    print(f"{name} state timeout")
    return False


def main():
    rclpy.init()

    ur5e = GazeboUR5eAdapter()
    panda = GazeboPandaAdapter()

    runtime = RobotRuntime()

    try:
        print("\nWaiting for robot states...")

        ur_ready = wait_for_state(ur5e, "UR5e")
        panda_ready = wait_for_state(panda, "Panda")

        if not ur_ready or not panda_ready:
            print("Robot state not ready.")
            return

        print("\nCurrent TCP poses")

        print("UR5e :", ur5e.get_tcp_pose())
        print("Panda :", panda.get_tcp_pose())

        ur_skill = MoveTCP(
            position=np.array([0.02, 0.02, 0.02]),
            frame="workspace",
            duration=8.0,
            tolerance=0.01
        )

        panda_skill = MoveTCP(
            position=np.array([0.02, 0.02, 0.02]),
            frame="workspace",
            duration=8.0,
            tolerance=0.01
        )

        print("\n==============================")
        print("UR5e semantic MoveTCP")
        print("==============================")

        ur_result = runtime.execute(
            ur5e,
            ur_skill
        )

        print("success :", ur_result.success)
        print("target  :", ur_result.target)
        print("actual  :", ur_result.actual)
        print("error   :", ur_result.error)
        print("failure :", ur_result.failure_reason)

        print("\n==============================")
        print("Panda semantic MoveTCP")
        print("==============================")

        panda_result = runtime.execute(
            panda,
            panda_skill
        )

        print("success :", panda_result.success)
        print("target  :", panda_result.target)
        print("actual  :", panda_result.actual)
        print("error   :", panda_result.error)
        print("failure :", panda_result.failure_reason)

        print("\n==============================")
        print("SUMMARY")
        print("==============================")

        print("UR5e :", "PASS" if ur_result.success else "FAIL")
        print("Panda:", "PASS" if panda_result.success else "FAIL")

    finally:
        ur5e.destroy_node()
        panda.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
