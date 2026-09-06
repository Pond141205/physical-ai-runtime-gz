import rclpy

from physical_ai_runtime.perception.camera_manager import CameraManager
from physical_ai_runtime.perception.inference_manager import InferenceManager


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
            "════════ SNAPSHOT ACQUISITION ════════"
            f"{W}"
        )

        manager = CameraManager(
            query="cube"
        )

        snapshot = manager.capture_snapshot(
            timeout=5.0,
            max_cross_camera_skew_s=0.10,
            rgb_depth_tolerance_s=0.05,
        )

        if not snapshot.valid:
            print(
                f"{R}✗ SNAPSHOT INVALID:{W} "
                f"{snapshot.reason}"
            )
            return

        print(
            f"{G}✓ SNAPSHOT VALID{W}"
        )

        print(
            f"{Y}cross-camera skew:{W} "
            f"{snapshot.cross_camera_skew_s}"
        )

        for camera, item in (
            snapshot.cameras.items()
        ):
            print(
                f"▶ {camera.upper():5s} "
                f"timestamp="
                f"{item.observation_timestamp}"
            )

        print(
            f"\n{C}"
            "════════ SHARED INFERENCE ════════"
            f"{W}"
        )

        inference = InferenceManager()

        results = inference.detect_multiview(
            snapshot,
            query="cube",
        )

        for camera in (
            "main",
            "side",
            "wrist",
        ):
            item = results.get(camera)

            if item is None:
                print(
                    f"{R}▶ {camera.upper():5s} "
                    f"NO RESULT{W}"
                )
                continue

            print(
                f"{G}▶ {camera.upper():5s}{W} "
                f"detections="
                f"{len(item.detections)} "
                f"timestamp="
                f"{item.timestamp}"
            )

        print(
            f"\n{C}"
            "SNAPSHOT INFERENCE ONLY — "
            "NO GEOMETRY / NO MOTION"
            f"{W}"
        )

    finally:
        if manager is not None:
            manager.close()

        rclpy.shutdown()


if __name__ == "__main__":
    main()
