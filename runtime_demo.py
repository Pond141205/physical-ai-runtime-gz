import time
import numpy as np
import rclpy

from runtime import RobotRuntime
from skills import MoveTCP, TaskSequence
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


def make_sequence():
    return [
        MoveTCP(
            position=np.array([0.02, 0.02, 0.02]),
            frame="workspace",
            duration=8.0,
            tolerance=0.01
        ),
        MoveTCP(
            position=np.array([0.04, -0.02, 0.04]),
            frame="workspace",
            duration=8.0,
            tolerance=0.01
        ),
        MoveTCP(
            position=np.array([-0.02, 0.03, 0.01]),
            frame="workspace",
            duration=8.0,
            tolerance=0.01
        ),
    ]


def make_task():
    return TaskSequence(
        skills=[
            MoveTCP(
                position=np.array([0.02, 0.02, 0.02]),
                frame="workspace",
                duration=8.0,
                tolerance=0.01
            ),
            MoveTCP(
                position=np.array([0.04, -0.02, 0.04]),
                frame="workspace",
                duration=8.0,
                tolerance=0.01
            ),
            MoveTCP(
                position=np.array([-0.02, 0.03, 0.01]),
                frame="workspace",
                duration=8.0,
                tolerance=0.01
            ),
        ],
        stop_on_failure=True
    )


def run_task(runtime, adapter, name):

    print("\n========================================")
    print(f"{name} TASK SEQUENCE")
    print("========================================")

    task = make_task()

    result = runtime.execute_task(
        adapter,
        task
    )

    for i, step_result in enumerate(
        result.results,
        start=1
    ):
        print(f"\n--- Step {i} ---")
        print("success :", step_result.success)
        print("target  :", step_result.target)
        print("actual  :", step_result.actual)
        print("error   :", step_result.error)
        print("failure :", step_result.failure_reason)

    if not result.success:
        print(
            f"\n{name} task failed at step "
            f"{result.failed_step}"
        )

    return result


def sequence_passed(results):

    return (
        len(results) == 3
        and all(
            result.success
            for result in results
        )
    )


def main():

    rclpy.init()

    ur5e = GazeboUR5eAdapter()
    panda = GazeboPandaAdapter()

    runtime = RobotRuntime()

    try:

        print("\nWaiting for robot states...")

        ur_ready = wait_for_state(
            ur5e,
            "UR5e"
        )

        panda_ready = wait_for_state(
            panda,
            "Panda"
        )

        if not ur_ready or not panda_ready:
            print("Robot state not ready.")
            return

        print("\nCurrent TCP poses")

        print(
            "UR5e :",
            ur5e.get_tcp_pose()
        )

        print(
            "Panda:",
            panda.get_tcp_pose()
        )

        ur_result = run_task(
            runtime,
            ur5e,
            "UR5e"
        )

        panda_result = run_task(
            runtime,
            panda,
            "Panda"
        )

        ur_pass = ur_result.success
        panda_pass = panda_result.success

        print("\n========================================")
        print("MULTI-STEP SEMANTIC RUNTIME SUMMARY")
        print("========================================")

        print(
            "UR5e :",
            "PASS"
            if ur_pass
            else "FAIL"
        )

        print(
            "Panda:",
            "PASS"
            if panda_pass
            else "FAIL"
        )

        print(
            "Runtime:",
            "PASS"
            if ur_pass and panda_pass
            else "FAIL"
        )

    finally:

        ur5e.destroy_node()
        panda.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()
