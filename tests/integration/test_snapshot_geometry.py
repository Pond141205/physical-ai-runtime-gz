import rclpy

from physical_ai_runtime.perception.camera_manager import CameraManager
from physical_ai_runtime.perception.inference_manager import InferenceManager
from physical_ai_runtime.perception.snapshot_geometry_processor import (
    SnapshotGeometryProcessor,
)


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
            "════════ 1 — FROZEN SNAPSHOT ════════"
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
                f"{R}✗ SNAPSHOT INVALID "
                f"{snapshot.reason}{W}"
            )
            return

        print(
            f"{G}✓ SNAPSHOT VALID{W} "
            f"skew="
            f"{snapshot.cross_camera_skew_s:.6f}s"
        )

        print(
            f"\n{C}"
            "════════ 2 — SHARED INFERENCE ════════"
            f"{W}"
        )

        inference = InferenceManager()

        detections = (
            inference.detect_multiview(
                snapshot,
                query="cube",
            )
        )

        for camera in (
            "main",
            "side",
            "wrist",
        ):
            print(
                f"▶ {camera.upper():5s} "
                f"detections="
                f"{len(detections[camera].detections)}"
            )

        print(
            f"\n{C}"
            "════════ 3 — SNAPSHOT GEOMETRY ════════"
            f"{W}"
        )

        processor = SnapshotGeometryProcessor(
            query="cube"
        )

        geometry = (
            processor.process_multiview(
                manager,
                snapshot,
                detections,
            )
        )

        for camera in (
            "main",
            "side",
            "wrist",
        ):
            result = geometry[camera]

            color = (
                G
                if result.status
                == "OBJECT_VISIBLE"
                else Y
            )

            print(
                f"{color}▶ "
                f"{camera.upper():5s}{W} "
                f"{result.status}"
            )

            print(
                f"  metrics: "
                f"{result.metrics}"
            )

            if result.scene is not None:
                obj = result.scene.objects[
                    "cube"
                ]

                print(
                    f"  world: "
                    f"{obj.position_world}"
                )

                print(
                    f"  robot: "
                    f"{obj.position_robot}"
                )

        print(
            f"\n{C}"
            "FROZEN SNAPSHOT GEOMETRY ONLY — "
            "NO AI FUSION / NO MOTION"
            f"{W}"
        )

    finally:
        if manager is not None:
            manager.close()

        rclpy.shutdown()


if __name__ == "__main__":
    main()
