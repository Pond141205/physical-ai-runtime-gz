import rclpy

from gazebo_panda_adapter import GazeboPandaAdapter
from gazebo_runtime_adapter import GazeboUR5eAdapter
from runtime import RobotRuntime
from skills import TaskSequence, Release, Grasp


def run_panda(runtime):

    robot = GazeboPandaAdapter()

    task = TaskSequence(
        skills=[
            Release(width=0.08),
            Grasp(width=0.00),
        ],
        stop_on_failure=True
    )

    result = runtime.execute_task(robot, task)

    print("\nPANDA RESULT")
    print(result)

    passed = (
        result.success
        and result.failed_step is None
        and len(result.results) == 2
    )

    robot.destroy_node()

    return passed


def run_ur5e(runtime):

    robot = GazeboUR5eAdapter()

    task = TaskSequence(
        skills=[
            Release(width=0.08),
            Grasp(width=0.00),
        ],
        stop_on_failure=True
    )

    result = runtime.execute_task(robot, task)

    print("\nUR5E RESULT")
    print(result)

    passed = (
        not result.success
        and result.failed_step == 1
        and result.failure_reason
            == "UNSUPPORTED_CAPABILITY"
    )

    robot.destroy_node()

    return passed


def main():

    rclpy.init()

    runtime = RobotRuntime()

    panda_pass = run_panda(runtime)
    ur5e_pass = run_ur5e(runtime)

    print("\n========================================")
    print(" SEMANTIC GRIPPER REGRESSION SUMMARY")
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
