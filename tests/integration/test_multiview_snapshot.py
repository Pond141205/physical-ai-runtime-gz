import rclpy

from physical_ai_runtime.perception.camera_manager import CameraManager


C = "\033[1;36m"
G = "\033[1;32m"
Y = "\033[1;33m"
R = "\033[1;31m"
W = "\033[0m"


def main():
    rclpy.init()

    manager = None

    try:
        print(
            f"\n{C}"
            "════════ CONCURRENT RGB-D SNAPSHOT ════════"
            f"{W}"
        )

        manager = CameraManager(
            query="cube",
        )

        result = manager.snapshot_all(
            timeout=5.0,
            max_cross_camera_skew_s=0.10,
            rgb_depth_tolerance_s=0.05,
        )

        for camera in (
            "main",
            "side",
            "wrist",
        ):
            item = result[
                "camera_status"
            ].get(
                camera,
                {},
            )

            ok = item.get(
                "ready",
                False,
            )

            color = G if ok else R

            print(
                f"{color}▶ {camera.upper():5s}{W} "
                f"ready={ok} "
                f"rgb-depth Δt="
                f"{item.get('rgb_depth_delta_s')} "
                f"timestamp="
                f"{item.get('timestamp')} "
                f"reason="
                f"{item.get('reason')}"
            )

        skew = result.get(
            "cross_camera_skew_s"
        )

        print(
            f"\n{Y}cross-camera skew:{W} "
            f"{skew}"
        )

        if result.get("valid"):
            print(
                f"{G}✓ SNAPSHOT_VALID{W}"
            )
        else:
            print(
                f"{R}✗ SNAPSHOT_INVALID{W} "
                f"reason={result.get('reason')}"
            )

        print(
            f"\n{C}"
            "SENSOR SNAPSHOT ONLY — "
            "NO INFERENCE / NO MOTION"
            f"{W}"
        )

    finally:
        if manager is not None:
            manager.close()

        rclpy.shutdown()


if __name__ == "__main__":
    main()
