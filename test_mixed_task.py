import time
import numpy as np
import rclpy

from runtime import RobotRuntime
from skills import MoveTCP, MoveJoints, TaskSequence
from gazebo_runtime_adapter import GazeboUR5eAdapter
from gazebo_panda_adapter import GazeboPandaAdapter


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


def build_task(adapter):

    state = adapter.get_state()

    current_q = state.joint_position.copy()

    # Small relative joint motion to keep test safe.
    joint_target = current_q.copy()
    joint_target[0] += 0.03

    return TaskSequence(
        skills=[
            MoveTCP(
                position=np.array([
                    0.02,
                    0.02,
                    0.02
                ]),
                frame="workspace",
                duration=8.0,
                tolerance=0.01
            ),

            MoveJoints(
                positions=joint_target,
                duration=5.0
            ),

            MoveTCP(
                position=np.array([
                    0.04,
                    -0.02,
                    0.04
                ]),
                frame="workspace",
                duration=8.0,
                tolerance=0.01
            ),
        ],
        stop_on_failure=True
    )


def print_result(name, result):

    print("\n========================================")
    print(f"{name} MIXED TASK RESULT")
    print("========================================")

    for i, step in enumerate(
        result.results,
        start=1
    ):
        print(f"\n--- Step {i} ---")
        print("success :", step.success)
        print("target  :", step.target)
        print("actual  :", step.actual)
        print("error   :", step.error)
        print("failure :", step.failure_reason)

    print("\nTask success :", result.success)
    print("Failed step  :", result.failed_step)
    print("Failure      :", result.failure_reason)


def main():

    rclpy.init()

    ur5e = GazeboUR5eAdapter()
    panda = GazeboPandaAdapter()

    runtime = RobotRuntime()

    try:

        print("\nWaiting for robot states...")

        if not wait_for_state(
            ur5e,
            "UR5e"
        ):
            return

        if not wait_for_state(
            panda,
            "Panda"
        ):
            return

        ur_task = build_task(
            ur5e
        )

        panda_task = build_task(
            panda
        )

        print("\nExecuting UR5e mixed semantic task...")

        ur_result = runtime.execute_task(
            ur5e,
            ur_task
        )

        print_result(
            "UR5e",
            ur_result
        )

        print("\nExecuting Panda mixed semantic task...")

        panda_result = runtime.execute_task(
            panda,
            panda_task
        )

        print_result(
            "Panda",
            panda_result
        )

        print("\n========================================")
        print("MIXED SEMANTIC TASK SUMMARY")
        print("========================================")

        print(
            "UR5e :",
            "PASS"
            if ur_result.success
            else "FAIL"
        )

        print(
            "Panda:",
            "PASS"
            if panda_result.success
            else "FAIL"
        )

        print(
            "Runtime:",
            "PASS"
            if (
                ur_result.success
                and panda_result.success
            )
            else "FAIL"
        )

    finally:

        ur5e.destroy_node()
        panda.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()
