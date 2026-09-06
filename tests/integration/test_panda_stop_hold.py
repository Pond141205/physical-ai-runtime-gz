"""Verify that Panda stop() holds the current controller command."""

import time

import numpy as np
import rclpy

from physical_ai_runtime.adapters.gazebo_panda_adapter import GazeboPandaAdapter


def main():
    rclpy.init()
    robot = GazeboPandaAdapter()

    try:
        before, _ = robot.get_tcp_pose(timeout=10.0)
        result = robot.stop()

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            rclpy.spin_once(robot, timeout_sec=0.05)

        after, _ = robot.get_tcp_pose(timeout=3.0)
        displacement = float(np.linalg.norm(after - before))

        print("STOP_HOLD_RESULT", result.success)
        print("TCP_BEFORE", before.tolist())
        print("TCP_AFTER", after.tolist())
        print("TCP_DISPLACEMENT", f"{displacement:.6f}")

        assert result.success
        assert result.skill == "STOP_HOLD"
        assert displacement <= 0.002
    finally:
        robot.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
