import os
import time

import numpy as np
import rclpy

from gazebo_panda_adapter import GazeboPandaAdapter
from runtime import RobotRuntime
from skills import MoveTCP


TARGET_TCP = np.array([
    0.45,
    0.15,
    0.45,
], dtype=float)

TOLERANCE = 0.02
RUNS = int(
    os.environ.get(
        "PANDA_REPEATED_RUNS",
        "20",
    )
)


def fmt(vector):
    array = np.asarray(
        vector,
        dtype=float,
    )

    if array.size == 0:
        return "[]"

    return "[" + ", ".join(
        f"{value:.6f}"
        for value in array.tolist()
    ) + "]"


def spin_for(
    node,
    seconds,
):
    end = time.time() + seconds

    while rclpy.ok() and time.time() < end:
        rclpy.spin_once(
            node,
            timeout_sec=0.05,
        )


def result_timeout(result):
    reason = getattr(
        result,
        "failure_reason",
        None,
    )

    return bool(
        reason
        and "TIMEOUT" in str(reason)
    )


def run_repeated_move_tcp(
    runs=RUNS,
):
    rclpy.init()

    robot = GazeboPandaAdapter()
    runtime = RobotRuntime()

    records = []

    try:
        robot.wait_for_joint_state(
            timeout=10.0,
        )

        initial_tcp, _ = robot.get_tcp_pose(
            timeout=10.0,
        )

        print(
            "PANDA_REPEATED_RUN_START "
            f"target_tcp={fmt(TARGET_TCP)} "
            f"initial_tcp={fmt(initial_tcp)} "
            f"tolerance={TOLERANCE:.6f} "
            f"runs={runs}"
        )

        for run_number in range(1, runs + 1):
            home_result = robot.go_home(
                duration=3.0,
            )

            spin_for(
                robot,
                0.4,
            )

            result = runtime.execute(
                robot,
                MoveTCP(
                    position=TARGET_TCP,
                    frame="robot_base",
                    tolerance=TOLERANCE,
                    timeout=5.0,
                    duration=3.0,
                ),
            )

            actual_tcp, _ = robot.get_tcp_pose(
                timeout=3.0,
            )

            error = float(
                np.linalg.norm(
                    TARGET_TCP - actual_tcp,
                )
            )

            timed_out = result_timeout(
                result,
            )

            result_success = bool(
                getattr(
                    result,
                    "success",
                    False,
                )
            )

            passed = (
                result_success
                and error <= TOLERANCE
                and not timed_out
            )

            records.append({
                "run": run_number,
                "error": error,
                "timeout": timed_out,
                "passed": passed,
            })

            print(
                "PANDA_REPEATED_RUN "
                f"run={run_number} "
                f"target_tcp={fmt(TARGET_TCP)} "
                f"actual_tcp={fmt(actual_tcp)} "
                f"euclidean_error={error:.6f} "
                f"timeout={timed_out} "
                "convergence_iterations=NA "
                f"convergence_time={float(result.duration):.3f} "
                f"success={result_success} "
                f"failure={result.failure_reason} "
                f"home_success={bool(home_result.success)} "
                f"home_error={float(home_result.error):.6f}"
            )

        errors = [
            record["error"]
            for record in records
        ]

        timeout_count = sum(
            1
            for record in records
            if record["timeout"]
        )

        failures = sum(
            1
            for record in records
            if not record["passed"]
        )

        max_error = max(errors)
        mean_error = float(
            np.mean(errors)
        )

        print(
            "PANDA_REPEATED_RUN_SUMMARY "
            f"passed={runs - failures}/{runs} "
            f"max_error={max_error:.6f} "
            f"mean_error={mean_error:.6f} "
            f"timeout_count={timeout_count} "
            f"failures={failures}"
        )

        assert max_error <= TOLERANCE
        assert timeout_count == 0
        assert failures == 0

        return records

    finally:
        robot.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


def test_panda_repeated_move_tcp():
    run_repeated_move_tcp()


if __name__ == "__main__":
    run_repeated_move_tcp()
