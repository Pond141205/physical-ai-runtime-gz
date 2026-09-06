import time
import rclpy
import numpy as np

from physical_ai_runtime.adapters.gazebo_panda_adapter import (
    GazeboPandaAdapter,
)

from physical_ai_runtime.planning.viewpoint_plan_validator import (
    ViewpointPlanValidator,
)


C = "\033[1;36m"
G = "\033[1;32m"
R = "\033[1;31m"
Y = "\033[1;33m"
W = "\033[0m"


def wait_for_tf(
    robot,
    target_frame,
    source_frame,
    timeout_s=5.0,
):
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        rclpy.spin_once(
            robot,
            timeout_sec=0.1,
        )

        try:
            tf = robot.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
            )

            return tf

        except Exception:
            pass

    return None


def main():
    rclpy.init()

    robot = GazeboPandaAdapter()

    try:
        print(
            f"\n{C}"
            "════════ WAIT FOR PANDA TF ════════"
            f"{W}"
        )

        tf = wait_for_tf(
            robot,
            robot.base_frame,
            "world",
            timeout_s=5.0,
        )

        if tf is None:
            print(
                f"{R}"
                "✗ TF panda_link0 <- world unavailable"
                f"{W}"
            )
            raise SystemExit(1)

        print(
            f"{G}"
            "✓ TF READY: world → panda_link0"
            f"{W}"
        )

        validator = (
            ViewpointPlanValidator(
                robot
            )
        )

        world_position = np.array([
            0.457813,
            -0.359746,
            1.048553,
        ])

        world_quaternion = np.array([
            0.0,
            0.367738,
            0.0,
            0.929929,
        ])

        result = (
            validator
            ._world_hand_to_base(
                world_position,
                world_quaternion,
            )
        )

        print(
            f"\n{C}"
            "════════ WORLD → BASE POSE ════════"
            f"{W}"
        )

        print(
            f"{Y}success:{W}",
            result["success"],
        )

        if not result["success"]:
            print(
                f"{R}failure:{W}",
                result["failure_reason"],
            )
            raise SystemExit(1)

        print(
            f"{Y}position base:{W}",
            result[
                "position"
            ].round(6).tolist(),
        )

        print(
            f"{Y}orientation base:{W}",
            result[
                "orientation"
            ].round(6).tolist(),
        )

        expected = np.array([
            world_position[0] - 0.6,
            world_position[1],
            world_position[2],
        ])

        error = float(
            np.linalg.norm(
                result["position"]
                - expected
            )
        )

        print(
            f"{Y}position error:{W}",
            error,
        )

        assert error < 1e-5

        print(
            f"\n{G}"
            "✓ WORLD → PANDA BASE TF PASSED"
            f"{W}"
        )

        print(
            "\nTF ONLY — "
            "NO MOVEIT PLAN — "
            "NO ROBOT MOTION"
        )

    finally:
        robot.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
