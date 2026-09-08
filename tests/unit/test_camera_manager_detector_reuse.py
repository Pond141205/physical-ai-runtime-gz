import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from physical_ai_runtime.perception.object_tracker import (
    ContinuousPerceptionFrame,
)
from physical_ai_runtime.perception.camera_manager import CameraManager


class CameraManagerDetectorReuseTest(unittest.TestCase):
    @staticmethod
    def _manager():
        manager = CameraManager.__new__(CameraManager)
        manager._inference_manager = None
        manager.detector = None
        manager._closed = False
        manager._grasp_observation_lock = threading.Lock()
        manager._inference_init_lock = threading.Lock()
        manager._executor_started = False
        manager._continuous_query = None
        manager._continuous_thread = None
        manager._continuous_cache_max_age_s = 0.75
        manager.start_sensor_streams = Mock()
        manager.main_observer = SimpleNamespace(detector=None)
        manager.side_observer = SimpleNamespace(
            detector=None,
            query="cube",
            reference_world=None,
            reference_size=None,
            workspace=None,
            self_mask_overlap_reject=0.35,
            observe_once=Mock(return_value="scene"),
        )
        manager.wrist_observer = SimpleNamespace(detector=None)
        return manager

    def test_shared_detector_is_attached_once_per_manager(self):
        manager = self._manager()
        shared_detector = object()
        inference = SimpleNamespace(detector=shared_detector)

        with patch(
            "physical_ai_runtime.perception.camera_manager.InferenceManager",
            return_value=inference,
        ) as factory:
            manager._ensure_shared_detector()
            manager._ensure_shared_detector()

        self.assertEqual(factory.call_count, 1)
        self.assertIs(manager.detector, shared_detector)
        self.assertIs(manager.main_observer.detector, shared_detector)
        self.assertIs(manager.side_observer.detector, shared_detector)
        self.assertIs(manager.wrist_observer.detector, shared_detector)

    def test_grasp_recheck_uses_persistent_observer_and_fresh_call(self):
        manager = self._manager()
        shared_detector = object()
        manager._inference_manager = SimpleNamespace(
            detector=shared_detector,
        )

        scene = manager.observe_grasp_scene(
            query="cube",
            reference_world=[1.0, 2.0, 3.0],
            reference_size=[0.04, 0.04, 0.04],
            workspace={"x_min": 0.0},
            timeout=1.5,
        )

        self.assertEqual(scene, "scene")
        manager.side_observer.observe_once.assert_called_once_with(
            timeout=1.5,
            spin=False,
        )
        self.assertIs(
            manager.side_observer.detector,
            shared_detector,
        )
        self.assertEqual(manager.side_observer.query, "cube")
        self.assertIsNone(manager.side_observer.reference_world)
        self.assertIsNone(manager.side_observer.reference_size)
        self.assertIsNone(manager.side_observer.workspace)
        self.assertEqual(
            manager.side_observer.self_mask_overlap_reject,
            0.35,
        )

    def test_continuous_worker_publishes_fresh_tracks(self):
        manager = self._manager()
        manager._continuous_state_lock = threading.Lock()
        manager._continuous_stop = threading.Event()
        manager._continuous_thread = None
        manager._continuous_query = None
        manager._continuous_rate_hz = 100.0
        manager._continuous_snapshot_timeout = 0.01
        manager._continuous_cache_max_age_s = 1.0
        manager._continuous_trackers = {}
        manager._continuous_frame = None

        snapshot = SimpleNamespace(
            valid=True,
            reason="SNAPSHOT_VALID",
            cameras={
                "main": SimpleNamespace(
                    observation_timestamp=1.0,
                ),
            },
        )
        inference = SimpleNamespace(
            timestamp=1.0,
            detections=[
                SimpleNamespace(
                    object_id="cube_0",
                    label="cube",
                    confidence=0.9,
                    bbox=(10, 10, 30, 30),
                ),
            ],
        )
        manager.capture_snapshot = Mock(return_value=snapshot)
        manager._inference_manager = SimpleNamespace(
            detect_multiview=Mock(
                return_value={"main": inference},
            ),
        )
        manager._ensure_shared_detector = Mock()

        manager.start_continuous_perception(
            query="cube",
            rate_hz=100.0,
            snapshot_timeout=0.01,
            cache_max_age_s=1.0,
        )

        deadline = time.monotonic() + 1.0
        frame = None
        while time.monotonic() < deadline:
            frame = manager.get_latest_perception(
                query="cube",
                max_age_s=1.0,
            )
            if frame is not None:
                break
            time.sleep(0.005)

        manager.stop_continuous_perception()

        self.assertIsNotNone(frame)
        self.assertEqual(
            frame.reason,
            "CONTINUOUS_PERCEPTION_READY",
        )
        self.assertEqual(
            frame.cameras["main"][0].label,
            "cube",
        )
        self.assertEqual(
            frame.cameras["main"][0].state,
            "DETECTED",
        )

    def test_fresh_continuous_cache_avoids_duplicate_detector_call(self):
        manager = self._manager()
        manager._continuous_state_lock = threading.Lock()
        manager._continuous_thread = SimpleNamespace(
            is_alive=lambda: True,
        )
        manager._continuous_cache_max_age_s = 1.0
        manager._continuous_frame = ContinuousPerceptionFrame(
            query="cube",
            observation_timestamp=1.0,
            received_monotonic=time.monotonic(),
            cameras={},
            inference_results={"main": object()},
            snapshot=SimpleNamespace(valid=True),
            detector_latency_s=0.2,
            valid=True,
            reason="CONTINUOUS_PERCEPTION_READY",
        )
        manager._ensure_shared_detector = Mock()
        manager._geometry_processor = Mock()
        manager._geometry_processor.process_multiview = Mock(
            return_value={}
        )
        manager.capture_snapshot = Mock()
        manager._inference_manager = SimpleNamespace(
            detect_multiview=Mock(),
        )

        result = manager.observe_manipulation_target(
            query="cube",
            timeout=1.0,
        )

        self.assertFalse(result["success"])
        manager.capture_snapshot.assert_not_called()
        manager._inference_manager.detect_multiview.assert_not_called()


if __name__ == "__main__":
    unittest.main()
