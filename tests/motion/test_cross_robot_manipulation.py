import rclpy
import numpy as np

from physical_ai_runtime.adapters.gazebo_panda_adapter import GazeboPandaAdapter
from physical_ai_runtime.adapters.gazebo_runtime_adapter import GazeboUR5eAdapter
from physical_ai_runtime.core.runtime import RobotRuntime
from physical_ai_runtime.core.skills import TaskSequence, MoveTCP, Release, Grasp


def build_task():
    return TaskSequence(
        skills=[
            MoveTCP(
                position=np.array([0.02, 0.02, 0.02]),
                frame="workspace",
                duration=3.0
            ),
            Release(width=0.08),
            Grasp(width=0.00),
            MoveTCP(
                position=np.array([0.00, 0.00, 0.05]),
                frame="workspace",
                duration=3.0
            ),
        ],
        stop_on_failure=True
    )


def run_panda(runtime):

    robot = GazeboPandaAdapter()
    result = runtime.execute_task(
        robot,
        build_task()
    )

    print("\nPANDA RESULT")
    print(result)

    passed = (
        result.success
        and result.failed_step is None
        and len(result.results) == 4
    )

    robot.destroy_node()
    return passed


def run_ur5e(runtime):

    robot = GazeboUR5eAdapter()
    result = runtime.execute_task(
        robot,
        build_task()
    )

    print("\nUR5E RESULT")
    print(result)

    passed = (
        not result.success
        and result.failed_step == 2
        and result.failure_reason
            == "UNSUPPORTED_CAPABILITY"
        and len(result.results) == 2
    )

    robot.destroy_node()
    return passed


def main():

    rclpy.init()

    runtime = RobotRuntime()

    panda_pass = run_panda(runtime)
    ur5e_pass = run_ur5e(runtime)

    print("\n========================================")
    print(" CROSS-ROBOT MANIPULATION SUMMARY")
    print("========================================")
    print(
        "Panda :",
        "PASS" if panda_pass else "FAIL"
    )
    print(
        "UR5e  :",
        "PASS" if ur5e_pass else "FAIL"
    )
    print(
        "Runtime:",
        "PASS"
        if panda_pass and ur5e_pass
        else "FAIL"
    )
    print("========================================")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
