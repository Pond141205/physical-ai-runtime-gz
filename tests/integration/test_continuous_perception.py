import os
import time

import rclpy

from physical_ai_runtime.perception.camera_manager import CameraManager


def main():
    query = os.getenv(
        "PHYSICAL_AI_CONTINUOUS_QUERY",
        "cube",
    )
    timeout_s = float(
        os.getenv(
            "PHYSICAL_AI_CONTINUOUS_TIMEOUT_S",
            "30.0",
        )
    )

    rclpy.init()
    manager = None

    try:
        print(
            "CONTINUOUS_PERCEPTION query="
            + query
        )
        manager = CameraManager(query=query)
        manager.start_continuous_perception(
            query=query,
            rate_hz=2.0,
            snapshot_timeout=1.0,
            cache_max_age_s=2.0,
        )

        deadline = time.monotonic() + timeout_s
        frame = None
        while time.monotonic() < deadline:
            frame = manager.get_latest_perception(
                query=query,
                max_age_s=2.0,
            )
            if frame is not None:
                break
            time.sleep(0.1)

        if frame is None:
            status = manager.get_continuous_perception_status()
            raise AssertionError(
                "CONTINUOUS_PERCEPTION_NOT_READY:"
                + str(
                    None
                    if status is None
                    else status.reason
                )
            )

        tracks = [
            track
            for camera_tracks in frame.cameras.values()
            for track in camera_tracks
            if track.label == query.casefold()
        ]

        if not tracks:
            raise AssertionError(
                "CONTINUOUS_LABEL_NOT_TRACKED:"
                + query
            )

        print(
            "CONTINUOUS_PERCEPTION=PASS"
        )
        print(
            "detector_latency_s="
            + str(frame.detector_latency_s)
        )
        print(
            "observation_timestamp="
            + str(frame.observation_timestamp)
        )
        print(
            "tracks="
            + str(
                [
                    {
                        "camera": camera,
                        "track_id": track.track_id,
                        "label": track.label,
                        "state": track.state,
                        "confidence": track.confidence,
                    }
                    for camera, camera_tracks in frame.cameras.items()
                    for track in camera_tracks
                ]
            )
        )
        print(
            "PERCEPTION ONLY - NO ROBOT MOTION"
        )
    finally:
        if manager is not None:
            manager.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
