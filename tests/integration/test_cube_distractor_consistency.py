"""Live perception regression for the launch-owned cube+distractor scene.

Run only after the canonical full stack is healthy with:
    PHYSICAL_AI_SCENE_VARIANT=cube_with_distractors

This test performs no robot motion and does not call a VLM.  It verifies that
deterministic cross-view geometry associates the semantic cube across cameras
instead of selecting the nearby cylinder distractor.
"""

import math

import numpy as np
import rclpy

from physical_ai_runtime.perception.camera_manager import CameraManager
from physical_ai_runtime.perception.inference_manager import InferenceManager
from physical_ai_runtime.perception.snapshot_geometry_processor import (
    SnapshotGeometryProcessor,
)


CUBE_WORLD = np.array([0.85, -0.35, 0.755], dtype=float)
CYLINDER_WORLD = np.array([0.76, -0.54, 0.755], dtype=float)
MAX_CUBE_POSITION_ERROR_M = 0.05
MAX_CROSS_VIEW_ERROR_M = 0.05
MIN_CYLINDER_SEPARATION_M = 0.12
MAX_ATTEMPTS = 3


def _distance(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def _visible_world(result):
    if result.status != "OBJECT_VISIBLE" or result.scene is None:
        return None

    cube = result.scene.objects.get("cube")
    return None if cube is None else cube.position_world


def main():
    rclpy.init()
    manager = None

    try:
        manager = CameraManager(query="cube")
        inference = InferenceManager()
        processor = SnapshotGeometryProcessor(query="cube")

        for attempt in range(1, MAX_ATTEMPTS + 1):
            snapshot = manager.capture_snapshot(
                timeout=5.0,
                max_cross_camera_skew_s=0.10,
                rgb_depth_tolerance_s=0.05,
            )

            if not snapshot.valid:
                print(
                    f"ATTEMPT={attempt} SNAPSHOT_INVALID "
                    f"reason={snapshot.reason}"
                )
                continue

            detections = inference.detect_multiview(snapshot, query="cube")
            geometry = processor.process_multiview(
                manager,
                snapshot,
                detections,
            )

            main_world = _visible_world(geometry["main"])
            side_world = _visible_world(geometry["side"])

            if main_world is None or side_world is None:
                print(
                    f"ATTEMPT={attempt} GEOMETRY_NOT_READY "
                    f"main={geometry['main'].status} "
                    f"side={geometry['side'].status}"
                )
                continue

            main_error = _distance(main_world, CUBE_WORLD)
            side_error = _distance(side_world, CUBE_WORLD)
            cross_view_error = _distance(main_world, side_world)
            side_to_cylinder = _distance(side_world, CYLINDER_WORLD)
            reference_error = geometry["side"].metrics.get(
                "reference_distance_m"
            )

            print(f"ATTEMPT={attempt}")
            print(f"MAIN_OBJECT_WORLD={main_world.tolist()}")
            print(f"SIDE_OBJECT_WORLD={side_world.tolist()}")
            print(f"MAIN_CUBE_ERROR_M={main_error:.6f}")
            print(f"SIDE_CUBE_ERROR_M={side_error:.6f}")
            print(f"CROSS_VIEW_ERROR_M={cross_view_error:.6f}")
            print(f"SIDE_REFERENCE_ERROR_M={reference_error:.6f}")
            print(f"SIDE_TO_CYLINDER_M={side_to_cylinder:.6f}")

            assert main_error <= MAX_CUBE_POSITION_ERROR_M
            assert side_error <= MAX_CUBE_POSITION_ERROR_M
            assert cross_view_error <= MAX_CROSS_VIEW_ERROR_M
            assert reference_error <= MAX_CROSS_VIEW_ERROR_M
            assert side_to_cylinder >= MIN_CYLINDER_SEPARATION_M
            print("CUBE_DISTRACTOR_CONSISTENCY=PASS")
            return

        raise AssertionError("CUBE_DISTRACTOR_CONSISTENCY=FAIL")
    finally:
        if manager is not None:
            manager.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
