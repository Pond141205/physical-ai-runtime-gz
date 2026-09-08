import threading
import time
from rclpy.executors import MultiThreadedExecutor
from dataclasses import dataclass
from typing import Optional, Any

from physical_ai_runtime.perception.rgbd_scene_observer import RGBDSceneObserver
from physical_ai_runtime.perception.grasp_verify_scene_observer import GraspVerifySceneObserver
from physical_ai_runtime.perception.wrist_rgbd_scene_observer import WristRGBDSceneObserver
from physical_ai_runtime.perception.shared_detector import SharedOpenVocabularyDetector
from physical_ai_runtime.perception.perception_snapshot import CameraSnapshot, MultiViewSnapshot
from physical_ai_runtime.perception.perception_uncertainty import FallbackDepthUncertaintyModel
from physical_ai_runtime.perception.inference_manager import InferenceManager
from physical_ai_runtime.perception.snapshot_geometry_processor import SnapshotGeometryProcessor
from physical_ai_runtime.perception.evidence_fusion import EvidenceFusion
from physical_ai_runtime.perception.object_tracker import (
    ContinuousPerceptionFrame,
    ObjectTracker,
)
from physical_ai_runtime.planning.viewpoint_selector import ViewpointSelector
from physical_ai_runtime.ai.agent.multiview_vision_reasoner import MultiviewVisionReasoner
from PIL import Image
from pathlib import Path


@dataclass
class CameraObservation:
    camera: str
    scene: Optional[Any]
    status: Optional[str]
    metrics: dict


class CameraManager:
    """
    Selects the best currently usable perception source.

    This layer does not command robot motion.
    It only evaluates available camera observations and
    exposes standardized visibility semantics.
    """

    def __init__(
        self,
        query="cube",
        robot_description_node="/panda/robot_state_publisher",
    ):
        self.query = query
        self.robot_description_node = robot_description_node

        self.last_selected_camera = None
        self.last_results = []

        # Detector is intentionally lazy.
        # Pure sensor acquisition must not allocate/load the
        # perception model or GPU resources.
        self.detector = None

        # Persistent observer nodes.
        # Sensor subscriptions and TF buffers survive across
        # repeated perception requests.
        self.main_observer = RGBDSceneObserver(
            query=self.query,
            robot_description_node=self.robot_description_node,
            detector=None,
            initialize_detector=False,
        )

        self.side_observer = GraspVerifySceneObserver(
            query=self.query,
            reference_world=None,
            reference_size=None,
            workspace=None,
            robot_description_node=self.robot_description_node,
            detector=None,
            initialize_detector=False,
        )

        self.wrist_observer = WristRGBDSceneObserver(
            query=self.query,
            robot_description_node=self.robot_description_node,
            detector=None,
            initialize_detector=False,
        )

        self._closed = False

        # Background ROS callback executor.
        # Camera callbacks run continuously and independently from
        # detector / AI inference.
        self._executor = None
        self._executor_thread = None
        self._executor_started = False
        self._grasp_observation_lock = threading.Lock()
        self._inference_init_lock = threading.Lock()

        # Continuous detect-then-track state. Sensor callbacks remain
        # independent from the inference worker and runtime motion.
        self._continuous_state_lock = threading.Lock()
        self._continuous_stop = threading.Event()
        self._continuous_thread = None
        self._continuous_query = None
        self._continuous_rate_hz = 2.0
        self._continuous_snapshot_timeout = 0.5
        self._continuous_cache_max_age_s = 0.75
        self._continuous_trackers = {}
        self._continuous_frame = None

        # Heavy perception components remain lazy.
        # Pure sensor snapshots do not load GPU or VLM resources.
        self._inference_manager = None
        self._geometry_processor = None
        self._vision_reasoner = None
        self._evidence_fusion = None
        self._viewpoint_selector = None

        # Replaceable by a calibrated camera-specific model
        # when running on real hardware.
        self.uncertainty_model = (
            FallbackDepthUncertaintyModel()
        )

    def _observe_main(self, timeout):
        observer = self.main_observer

        scene = observer.observe_once(
            timeout=timeout
        )

        return CameraObservation(
            camera="main",
            scene=scene,
            status=getattr(
                observer,
                "last_observation_status",
                None,
            ),
            metrics=dict(
                getattr(
                    observer,
                    "last_observation_metrics",
                    {},
                )
            ),
        )

    def _observe_wrist(self, timeout):
        observer = self.wrist_observer

        scene = observer.observe_once(
            timeout=timeout
        )

        return CameraObservation(
            camera="wrist",
            scene=scene,
            status=getattr(
                observer,
                "last_observation_status",
                None,
            ),
            metrics=dict(
                getattr(
                    observer,
                    "last_observation_metrics",
                    {},
                )
            ),
        )

    def observe_best(
        self,
        timeout_per_camera=4.0,
        include_main=True,
        include_wrist=True,
    ):
        self._ensure_shared_detector()
        results = []

        if include_main:
            results.append(
                self._observe_main(
                    timeout_per_camera
                )
            )

        if include_wrist:
            results.append(
                self._observe_wrist(
                    timeout_per_camera
                )
            )

        self.last_results = results

        visible = [
            result
            for result in results
            if (
                result.scene is not None
                and result.status == "OBJECT_VISIBLE"
                and self.query in result.scene.objects
            )
        ]

        if visible:
            #
            # Prefer a different view than the one most
            # recently used when several valid views exist.
            # This prevents immediately reusing the same
            # potentially problematic viewpoint.
            #
            for result in visible:
                if (
                    result.camera
                    != self.last_selected_camera
                ):
                    self.last_selected_camera = (
                        result.camera
                    )
                    return result

            chosen = visible[0]
            self.last_selected_camera = (
                chosen.camera
            )
            return chosen

        #
        # Distinguish robot self-occlusion from an actual
        # lack of semantic detection.
        #
        if any(
            result.status == "SELF_OCCLUDED"
            for result in results
        ):
            return CameraObservation(
                camera="none",
                scene=None,
                status="SELF_OCCLUDED",
                metrics={
                    "cameras": {
                        result.camera: {
                            "status":
                                result.status,
                            "metrics":
                                result.metrics,
                        }
                        for result in results
                    }
                },
            )

        return CameraObservation(
            camera="none",
            scene=None,
            status="OBJECT_NOT_DETECTED",
            metrics={
                "cameras": {
                    result.camera: {
                        "status":
                            result.status,
                        "metrics":
                            result.metrics,
                    }
                    for result in results
                }
            },
        )

    def observe_side(self, timeout=4.0):
        self._ensure_shared_detector()
        observer = self.side_observer

        scene = observer.observe_once(
            timeout=timeout
        )

        return CameraObservation(
            camera="side",
            scene=scene,
            status=getattr(
                observer,
                "last_observation_status",
                None,
            ),
            metrics=dict(
                getattr(
                    observer,
                    "last_observation_metrics",
                    {},
                )
            ),
        )

    def start_sensor_streams(self):
        """
        Start one background ROS executor for all persistent
        camera observer nodes.

        This only services subscriptions / TF callbacks.
        It does not run detector inference or robot motion.
        """
        if self._executor_started:
            return

        self._executor = MultiThreadedExecutor(
            num_threads=4
        )

        for observer in (
            self.main_observer,
            self.side_observer,
            self.wrist_observer,
        ):
            self._executor.add_node(observer)

        self._executor_thread = threading.Thread(
            target=self._executor.spin,
            name="physical-ai-camera-executor",
            daemon=True,
        )

        self._executor_thread.start()
        self._executor_started = True


    @staticmethod
    def _snapshot_ready(
        snapshot,
        rgb_depth_tolerance_s,
    ):
        if snapshot is None:
            return False, "NO_SNAPSHOT"

        required = (
            "rgb",
            "depth",
            "rgb_timestamp",
            "depth_timestamp",
            "observation_timestamp",
            "fx",
            "fy",
            "cx",
            "cy",
        )

        for key in required:
            if snapshot.get(key) is None:
                return False, f"MISSING_{key.upper()}"

        dt = abs(
            snapshot["rgb_timestamp"]
            - snapshot["depth_timestamp"]
        )

        if dt > rgb_depth_tolerance_s:
            return False, "RGB_DEPTH_TIME_MISMATCH"

        return True, "READY"


    def snapshot_all(
        self,
        timeout=2.0,
        max_cross_camera_skew_s=0.10,
        rgb_depth_tolerance_s=0.05,
    ):
        """
        Capture the latest coherent RGB-D state from all cameras.

        Safety properties:
        - RGB/depth pair must be synchronized per camera.
        - All camera observation timestamps must lie inside the
          requested cross-camera time window.
        - No detector inference.
        - No AI call.
        - No robot motion.

        Returns diagnostics even when invalid (fail closed).
        """
        self.start_sensor_streams()

        deadline = time.monotonic() + float(timeout)
        last_result = None

        observers = {
            "main": self.main_observer,
            "side": self.side_observer,
            "wrist": self.wrist_observer,
        }

        while time.monotonic() < deadline:
            snapshots = {
                name: observer.snapshot_sensor_state()
                for name, observer in observers.items()
            }

            camera_status = {}
            all_ready = True

            for name, snap in snapshots.items():
                ready, reason = self._snapshot_ready(
                    snap,
                    rgb_depth_tolerance_s,
                )

                rgb_depth_delta = None

                if (
                    snap.get("rgb_timestamp") is not None
                    and snap.get("depth_timestamp") is not None
                ):
                    rgb_depth_delta = abs(
                        snap["rgb_timestamp"]
                        - snap["depth_timestamp"]
                    )

                camera_status[name] = {
                    "ready": ready,
                    "reason": reason,
                    "rgb_depth_delta_s": rgb_depth_delta,
                    "timestamp": snap.get(
                        "observation_timestamp"
                    ),
                }

                if not ready:
                    all_ready = False

            cross_camera_skew = None
            timestamp_valid = False

            if all_ready:
                timestamps = [
                    snapshots[name]["observation_timestamp"]
                    for name in (
                        "main",
                        "side",
                        "wrist",
                    )
                ]

                cross_camera_skew = (
                    max(timestamps)
                    - min(timestamps)
                )

                timestamp_valid = (
                    cross_camera_skew
                    <= max_cross_camera_skew_s
                )

            snapshot_valid = (
                all_ready
                and timestamp_valid
            )

            last_result = {
                "valid": snapshot_valid,
                "snapshots": snapshots,
                "camera_status": camera_status,
                "cross_camera_skew_s": cross_camera_skew,
                "max_cross_camera_skew_s": (
                    max_cross_camera_skew_s
                ),
                "reason": (
                    "SNAPSHOT_VALID"
                    if snapshot_valid
                    else (
                        "CROSS_CAMERA_TIME_MISMATCH"
                        if all_ready
                        else "CAMERA_NOT_READY"
                    )
                ),
            }

            if snapshot_valid:
                return last_result

            time.sleep(0.02)

        return last_result or {
            "valid": False,
            "snapshots": {},
            "camera_status": {},
            "cross_camera_skew_s": None,
            "max_cross_camera_skew_s": (
                max_cross_camera_skew_s
            ),
            "reason": "SNAPSHOT_TIMEOUT",
        }


    def freeze_snapshot(self, result):
        """
        Convert snapshot_all() output into immutable perception input.

        The arrays returned by each observer are already copies,
        so inference cannot observe subsequent ROS callback updates.
        """
        if not result or not result.get("valid", False):
            return MultiViewSnapshot(
                cameras={},
                cross_camera_skew_s=(
                    None
                    if not result
                    else result.get(
                        "cross_camera_skew_s"
                    )
                ),
                valid=False,
                reason=(
                    "NO_SNAPSHOT"
                    if not result
                    else result.get(
                        "reason",
                        "INVALID_SNAPSHOT",
                    )
                ),
            )

        frozen = {}

        for camera, raw in result[
            "snapshots"
        ].items():

            frozen[camera] = CameraSnapshot(
                camera=camera,

                rgb=raw["rgb"],
                depth=raw["depth"],

                rgb_timestamp=float(
                    raw["rgb_timestamp"]
                ),
                depth_timestamp=float(
                    raw["depth_timestamp"]
                ),
                observation_timestamp=float(
                    raw["observation_timestamp"]
                ),

                fx=float(raw["fx"]),
                fy=float(raw["fy"]),
                cx=float(raw["cx"]),
                cy=float(raw["cy"]),

                camera_frame=str(
                    raw["camera_frame"]
                ),

                sensor_frame=raw.get(
                    "sensor_frame"
                ),
            )

        return MultiViewSnapshot(
            cameras=frozen,
            cross_camera_skew_s=float(
                result["cross_camera_skew_s"]
            ),
            valid=True,
            reason="SNAPSHOT_VALID",
        )


    def capture_snapshot(
        self,
        timeout=2.0,
        max_cross_camera_skew_s=0.10,
        rgb_depth_tolerance_s=0.05,
    ):
        result = self.snapshot_all(
            timeout=timeout,
            max_cross_camera_skew_s=(
                max_cross_camera_skew_s
            ),
            rgb_depth_tolerance_s=(
                rgb_depth_tolerance_s
            ),
        )

        return self.freeze_snapshot(
            result
        )


    def _ensure_perception_stack(self):
        """
        Lazily initialize heavy perception components.

        GroundingDINO is loaded exactly once per CameraManager.
        """
        self._ensure_shared_detector()

        if self._geometry_processor is None:
            self._geometry_processor = (
                SnapshotGeometryProcessor(
                    query=self.query
                )
            )

        if self._vision_reasoner is None:
            self._vision_reasoner = (
                MultiviewVisionReasoner()
            )

        if self._evidence_fusion is None:
            self._evidence_fusion = (
                EvidenceFusion()
            )

        if self._viewpoint_selector is None:
            self._viewpoint_selector = (
                ViewpointSelector()
            )


    def _ensure_shared_detector(self):
        """Attach one detector instance to every persistent observer."""
        with self._inference_init_lock:
            if self._inference_manager is None:
                self._inference_manager = InferenceManager()

            self.detector = self._inference_manager.detector

            for observer in (
                self.main_observer,
                self.side_observer,
                self.wrist_observer,
            ):
                observer.detector = self.detector

        return self.detector


    def start_continuous_perception(
        self,
        query=None,
        rate_hz=2.0,
        snapshot_timeout=0.5,
        cache_max_age_s=0.75,
    ):
        """Start background detector inference and object tracking."""
        if self._closed:
            raise RuntimeError("CAMERA_MANAGER_CLOSED")

        query = self.query if query is None else str(query).strip()
        if not query:
            raise ValueError("continuous perception query is required")
        if float(rate_hz) <= 0.0:
            raise ValueError("rate_hz must be positive")
        if float(snapshot_timeout) <= 0.0:
            raise ValueError("snapshot_timeout must be positive")
        if float(cache_max_age_s) <= 0.0:
            raise ValueError("cache_max_age_s must be positive")

        self.start_sensor_streams()

        with self._continuous_state_lock:
            self.query = query
            self._continuous_query = query
            self._continuous_rate_hz = float(rate_hz)
            self._continuous_snapshot_timeout = float(
                snapshot_timeout
            )
            self._continuous_cache_max_age_s = float(
                cache_max_age_s
            )

            if (
                self._continuous_thread is not None
                and self._continuous_thread.is_alive()
            ):
                return

            self._continuous_trackers = {}
            self._continuous_stop.clear()
            self._continuous_thread = threading.Thread(
                target=self._continuous_loop,
                name="physical-ai-continuous-perception",
                daemon=True,
            )
            self._continuous_thread.start()


    def set_continuous_query(self, query):
        """Change the tracked semantic query and reset stale tracks."""
        query = str(query).strip()
        if not query:
            raise ValueError("continuous perception query is required")

        with self._continuous_state_lock:
            if query != self._continuous_query:
                self._continuous_trackers = {}
                self._continuous_frame = None
            self.query = query
            self._continuous_query = query


    def _publish_continuous_frame(self, frame):
        with self._continuous_state_lock:
            self._continuous_frame = frame


    def _continuous_loop(self):
        try:
            self._ensure_shared_detector()
        except Exception as exc:
            self._publish_continuous_frame(
                ContinuousPerceptionFrame(
                    query=str(self._continuous_query),
                    observation_timestamp=None,
                    received_monotonic=time.monotonic(),
                    cameras={},
                    inference_results={},
                    snapshot=None,
                    detector_latency_s=None,
                    valid=False,
                    reason=(
                        "DETECTOR_INITIALIZATION_FAILED:"
                        + type(exc).__name__
                    ),
                )
            )
            return

        while not self._continuous_stop.is_set():
            cycle_started = time.monotonic()

            with self._continuous_state_lock:
                query = self._continuous_query
                rate_hz = self._continuous_rate_hz
                snapshot_timeout = self._continuous_snapshot_timeout

            try:
                snapshot = self.capture_snapshot(
                    timeout=snapshot_timeout
                )

                if snapshot is None or not snapshot.valid:
                    self._publish_continuous_frame(
                        ContinuousPerceptionFrame(
                            query=str(query),
                            observation_timestamp=None,
                            received_monotonic=time.monotonic(),
                            cameras={},
                            inference_results={},
                            snapshot=snapshot,
                            detector_latency_s=None,
                            valid=False,
                            reason=(
                                "INVALID_SENSOR_SNAPSHOT"
                                if snapshot is None
                                else snapshot.reason
                            ),
                        )
                    )
                else:
                    inference_started = time.monotonic()
                    inference_results = (
                        self._inference_manager.detect_multiview(
                            snapshot,
                            query=query,
                        )
                    )
                    detector_latency_s = (
                        time.monotonic() - inference_started
                    )

                    cameras = {}
                    for camera, result in inference_results.items():
                        tracker = self._continuous_trackers.get(camera)
                        if tracker is None:
                            tracker = ObjectTracker(camera=camera)
                            self._continuous_trackers[camera] = tracker
                        cameras[camera] = tracker.update(
                            result.detections,
                            result.timestamp,
                        )

                    self._publish_continuous_frame(
                        ContinuousPerceptionFrame(
                            query=str(query),
                            observation_timestamp=min(
                                camera_snapshot.observation_timestamp
                                for camera_snapshot in snapshot.cameras.values()
                            ),
                            received_monotonic=time.monotonic(),
                            cameras=cameras,
                            inference_results=dict(
                                inference_results
                            ),
                            snapshot=snapshot,
                            detector_latency_s=detector_latency_s,
                            valid=True,
                            reason="CONTINUOUS_PERCEPTION_READY",
                        )
                    )
            except Exception as exc:
                self._publish_continuous_frame(
                    ContinuousPerceptionFrame(
                        query=str(query),
                        observation_timestamp=None,
                        received_monotonic=time.monotonic(),
                        cameras={},
                        inference_results={},
                        snapshot=None,
                        detector_latency_s=None,
                        valid=False,
                        reason=(
                            "CONTINUOUS_PERCEPTION_ERROR:"
                            + type(exc).__name__
                        ),
                    )
                )

            period = 1.0 / max(0.1, float(rate_hz))
            self._continuous_stop.wait(
                timeout=max(
                    0.0,
                    period - (time.monotonic() - cycle_started),
                )
            )


    def get_latest_perception(self, query=None, max_age_s=None):
        """Return a fresh continuous frame, or None when unavailable."""
        with self._continuous_state_lock:
            frame = self._continuous_frame
            configured_age = self._continuous_cache_max_age_s
            thread = self._continuous_thread

        if (
            frame is None
            or not frame.valid
            or thread is None
            or not thread.is_alive()
        ):
            return None
        if query is not None and frame.query != str(query):
            return None

        max_age_s = (
            configured_age
            if max_age_s is None
            else float(max_age_s)
        )
        if max_age_s <= 0.0 or frame.age_s(time.monotonic()) > max_age_s:
            return None
        return frame


    def get_continuous_perception_status(self):
        with self._continuous_state_lock:
            return self._continuous_frame


    def stop_continuous_perception(self):
        with self._continuous_state_lock:
            thread = self._continuous_thread
            if thread is None:
                return
            self._continuous_stop.set()

        if thread is not threading.current_thread():
            thread.join(timeout=2.0)

        with self._continuous_state_lock:
            if self._continuous_thread is thread:
                self._continuous_thread = None


    def observe_grasp_scene(
        self,
        query,
        reference_world=None,
        reference_size=None,
        workspace=None,
        timeout=10.0,
        self_mask_overlap_reject=0.35,
    ):
        """Run fresh side-camera verification with persistent resources.

        Sensor callbacks, TF state and detector weights belong to this
        CameraManager session.  Each call still performs new detector
        inference over the latest RGB-D data; it only avoids rebuilding
        the ROS node and loading model weights again.
        """
        if self._closed:
            raise RuntimeError("CAMERA_MANAGER_CLOSED")

        self._ensure_shared_detector()
        self.start_sensor_streams()

        observer = self.side_observer

        with self._grasp_observation_lock:
            original = (
                observer.query,
                observer.reference_world,
                observer.reference_size,
                observer.workspace,
                observer.self_mask_overlap_reject,
            )

            observer.query = str(query)
            observer.reference_world = reference_world
            observer.reference_size = reference_size
            observer.workspace = workspace
            observer.self_mask_overlap_reject = float(
                self_mask_overlap_reject
            )

            try:
                return observer.observe_once(
                    timeout=timeout,
                    spin=False,
                )
            finally:
                (
                    observer.query,
                    observer.reference_world,
                    observer.reference_size,
                    observer.workspace,
                    observer.self_mask_overlap_reject,
                ) = original


    @staticmethod
    def _position_sigma_from_depth(depth):
        """
        Temporary fallback uncertainty model.

        Real cameras should replace this with calibrated
        sensor / depth uncertainty supplied by the adapter.
        """
        if depth is None:
            return None

        try:
            depth = float(depth)
        except (TypeError, ValueError):
            return None

        if depth <= 0.0:
            return None

        return min(
            0.10,
            max(
                0.008,
                0.008
                + 0.004 * depth * depth,
            ),
        )


    def _geometry_fusion_record(
        self,
        camera,
        result,
    ):
        metrics = dict(
            result.metrics or {}
        )

        depth = metrics.get("depth")
        world_xyz = metrics.get(
            "world_xyz"
        )

        if (
            world_xyz is None
            and result.scene is not None
        ):
            obj = result.scene.objects.get(
                "cube"
            )

            if obj is None and result.scene.objects:
                obj = next(
                    iter(
                        result.scene.objects.values()
                    )
                )

            if obj is not None:
                if obj.position_world is not None:
                    world_xyz = [
                        float(v)
                        for v in obj.position_world
                    ]

                if depth is None:
                    depth = obj.depth

        uncertainty = (
            self.uncertainty_model.estimate(
                camera=camera,
                depth_m=depth,
            )
        )

        sigma = uncertainty.sigma_m

        return {
            "status": result.status,
            "metrics": metrics,
            "scene": result.scene,

            "world_xyz": world_xyz,
            "depth": depth,

            "timestamp": metrics.get(
                "timestamp"
            ),

            "position_sigma_m": sigma,

            "uncertainty_source": (
                uncertainty.source
            ),

            # Fail closed until strict single-view validation
            # is implemented.
            "single_view_validated": False,
        }


    @staticmethod
    def _write_snapshot_images(
        snapshot,
        directory="/tmp/physical_ai_verified",
    ):
        """
        Export the exact frozen RGB arrays used by deterministic
        perception so the VLM receives the same observations.
        """
        output_dir = Path(directory)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        paths = {}

        for camera in (
            "main",
            "side",
            "wrist",
        ):
            item = snapshot.get(camera)

            if item is None:
                continue

            path = output_dir / (
                f"{camera}_"
                f"{int(item.observation_timestamp * 1e9)}"
                ".jpg"
            )

            Image.fromarray(
                item.rgb
            ).save(
                path,
                quality=95,
            )

            paths[camera] = str(path)

        return paths


    @staticmethod
    def _cleanup_snapshot_images(paths):
        for path in paths.values():
            try:
                Path(path).unlink(
                    missing_ok=True
                )
            except Exception:
                pass


    def observe_visual_scene(
        self,
        timeout=2.0,
        max_candidates=8,
    ):
        """
        Fast conversational vision.

        Uses:
          live synchronized RGB-D snapshot
          -> VLM visual interpretation

        Does NOT:
          load detector
          compute metric object geometry
          authorize motion
        """
        if self._closed:
            raise RuntimeError("CAMERA_MANAGER_CLOSED")

        snapshot = self.capture_snapshot(
            timeout=timeout,
        )

        if not snapshot.valid:
            return {
                "success": False,
                "reason": snapshot.reason,
                "scene_description": "",
                "candidates": [],
            }

        if self._vision_reasoner is None:
            self._vision_reasoner = (
                MultiviewVisionReasoner()
            )

        image_paths = self._write_snapshot_images(
            snapshot
        )

        try:
            discovery = (
                self._vision_reasoner
                .discover_objects(
                    image_paths,
                    max_candidates=max_candidates,
                )
            )
        finally:
            self._cleanup_snapshot_images(
                image_paths
            )

        return {
            "success": True,
            "reason": "VISUAL_SCENE_AVAILABLE",
            "scene_description":
                discovery.get(
                    "scene_description",
                    "",
                ),
            "candidates":
                discovery.get(
                    "candidates",
                    [],
                ),
        }


    def discover_scene(
        self,
        timeout=5.0,
        max_candidates=8,
        max_cross_camera_skew_s=0.10,
        rgb_depth_tolerance_s=0.05,
    ):
        """
        Discover a generic scene from one frozen multiview observation.

        Authority boundary:
        VLM -> semantic candidate proposal only
        Detector -> image grounding
        Depth/TF -> metric geometry

        Returned inventory is conversational/perception evidence.
        It is NOT authorization for robot motion.
        """
        if self._closed:
            raise RuntimeError(
                "CAMERA_MANAGER_CLOSED"
            )

        snapshot = self.capture_snapshot(
            timeout=timeout,
            max_cross_camera_skew_s=(
                max_cross_camera_skew_s
            ),
            rgb_depth_tolerance_s=(
                rgb_depth_tolerance_s
            ),
        )

        if not snapshot.valid:
            return {
                "success": False,
                "reason": "INVALID_SENSOR_SNAPSHOT",
                "scene_description": "",
                "candidates": [],
                "objects": [],
                "snapshot": snapshot,
            }

        self._ensure_perception_stack()

        image_paths = self._write_snapshot_images(
            snapshot
        )

        try:
            discovery = (
                self._vision_reasoner.discover_objects(
                    image_paths,
                    max_candidates=max_candidates,
                )
            )
        finally:
            self._cleanup_snapshot_images(
                image_paths
            )

        original_query = self.query
        objects = []

        try:
            for candidate in discovery[
                "candidates"
            ]:
                query = candidate["label"]

                self.query = query
                self._geometry_processor.query = query

                inference_results = (
                    self._inference_manager
                    .detect_multiview(
                        snapshot,
                        query=query,
                    )
                )

                geometry_results = (
                    self._geometry_processor
                    .process_multiview(
                        self,
                        snapshot,
                        inference_results,
                    )
                )

                verified_views = []
                positions = {}

                for camera, result in (
                    geometry_results.items()
                ):
                    if (
                        result.status
                        != "OBJECT_VISIBLE"
                        or result.scene is None
                        or query
                        not in result.scene.objects
                    ):
                        continue

                    obj = result.scene.objects[
                        query
                    ]

                    verified_views.append(camera)

                    if obj.position_world is not None:
                        positions[camera] = [
                            float(v)
                            for v
                            in obj.position_world
                        ]

                if not verified_views:
                    continue

                preferred_order = (
                    "main",
                    "side",
                    "wrist",
                )

                best_camera = next(
                    (
                        camera
                        for camera
                        in preferred_order
                        if camera in verified_views
                    ),
                    verified_views[0],
                )

                objects.append(
                    {
                        "label": query,
                        "description": candidate[
                            "description"
                        ],
                        "vlm_confidence": candidate[
                            "confidence"
                        ],
                        "vlm_cameras": candidate[
                            "cameras"
                        ],
                        "detector_verified": True,
                        "verified_cameras":
                            verified_views,
                        "best_camera": best_camera,
                        "position_world": (
                            positions.get(
                                best_camera
                            )
                        ),
                        "positions_world":
                            positions,
                    }
                )

        finally:
            self.query = original_query
            if self._geometry_processor is not None:
                self._geometry_processor.query = (
                    original_query
                )

        return {
            "success": True,
            "reason": "SCENE_DISCOVERY_COMPLETE",
            "scene_description": discovery[
                "scene_description"
            ],
            "candidates": discovery[
                "candidates"
            ],
            "objects": objects,
            "snapshot": snapshot,
        }



    def observe_manipulation_target(
        self,
        query=None,
        timeout=5.0,
        max_age_s=None,
    ):
        """
        Deterministic perception path for an explicitly named
        manipulation target.

        Pipeline:
            synchronized RGB-D
            -> GroundingDINO
            -> depth
            -> timestamped TF
            -> robot self-mask
            -> metric SceneState

        No VLM call is allowed here.
        """

        if query is not None:
            self.query = str(query)
            with self._continuous_state_lock:
                continuous_query = self._continuous_query
            if (
                continuous_query is not None
                and continuous_query != self.query
            ):
                self.set_continuous_query(self.query)

        cached = self.get_latest_perception(
            query=self.query,
            max_age_s=max_age_s,
        )

        if cached is not None:
            snapshot = cached.snapshot
            inference_results = cached.inference_results
        else:
            snapshot = self.capture_snapshot(
                timeout=timeout
            )
            inference_results = None

        if (
            snapshot is None
            or not snapshot.valid
        ):
            return {
                "success": False,
                "scene": None,
                "camera": None,
                "reason": (
                    "INVALID_SENSOR_SNAPSHOT"
                    if snapshot is None
                    else snapshot.reason
                ),
            }

        # Detector + geometry only.
        #
        # Do NOT call _ensure_perception_stack(), because that also
        # creates the cloud vision reasoner.
        self._ensure_shared_detector()

        if self._geometry_processor is None:
            self._geometry_processor = (
                SnapshotGeometryProcessor(
                    query=self.query
                )
            )

        self._geometry_processor.query = (
            self.query
        )

        if inference_results is None:
            inference_results = (
                self._inference_manager
                .detect_multiview(
                    snapshot,
                    query=self.query,
                )
            )

        geometry_results = (
            self._geometry_processor
            .process_multiview(
                self,
                snapshot,
                inference_results,
            )
        )

        # Prefer manipulation-complete geometry.
        # Main normally has support surface + object height.
        for camera in (
            "main",
            "side",
            "wrist",
        ):
            result = geometry_results.get(
                camera
            )

            if (
                result is None
                or result.scene is None
                or self.query
                not in result.scene.objects
            ):
                continue

            obj = result.scene.objects[
                self.query
            ]

            if (
                obj.position_robot is None
                or obj.support_z is None
                or obj.height is None
            ):
                continue

            return {
                "success": True,
                "scene": result.scene,
                "camera": camera,
                "reason":
                    "MANIPULATION_GEOMETRY_AVAILABLE",
                "snapshot": snapshot,
                "inference": inference_results,
                "geometry_results":
                    geometry_results,
            }

        return {
            "success": False,
            "scene": None,
            "camera": None,
            "reason":
                "MANIPULATION_GEOMETRY_UNAVAILABLE",
            "snapshot": snapshot,
            "inference": inference_results,
            "geometry_results":
                geometry_results,
        }


    def observe_verified(
        self,
        query=None,
        task=None,
        scene_context=None,
        timeout=5.0,
        max_cross_camera_skew_s=0.10,
        rgb_depth_tolerance_s=0.05,
    ):
        """
        Execute the complete perception-only Physical AI pipeline.

        Authority:
          sensor snapshot
            -> deterministic detector
            -> timestamped metric geometry
            -> multimodal visual evidence
            -> spatial/evidence fusion

        This method NEVER commands robot motion.
        """
        if self._closed:
            raise RuntimeError(
                "CAMERA_MANAGER_CLOSED"
            )

        if query is not None:
            self.query = query

        # --------------------------------------------------
        # 1. Freeze one synchronized RGB-D observation.
        # --------------------------------------------------

        snapshot = self.capture_snapshot(
            timeout=timeout,
            max_cross_camera_skew_s=(
                max_cross_camera_skew_s
            ),
            rgb_depth_tolerance_s=(
                rgb_depth_tolerance_s
            ),
        )

        if not snapshot.valid:
            return {
                "safe_visual_evidence": False,
                "scene": None,
                "best_camera": None,
                "snapshot": snapshot,
                "geometry": {},
                "vision": None,
                "fusion": None,
                "reason": (
                    "INVALID_SENSOR_SNAPSHOT"
                ),
            }

        self._ensure_perception_stack()

        # Keep processor query synchronized.
        self._geometry_processor.query = (
            self.query
        )

        # --------------------------------------------------
        # 2. Detector inference from FROZEN images.
        # --------------------------------------------------

        inference_results = (
            self._inference_manager
            .detect_multiview(
                snapshot,
                query=self.query,
            )
        )

        # --------------------------------------------------
        # 3. Metric geometry immediately after detection.
        #
        # Historical TF is consumed before any network/VLM
        # latency can age the TF buffer.
        # --------------------------------------------------

        geometry_results = (
            self._geometry_processor
            .process_multiview(
                self,
                snapshot,
                inference_results,
            )
        )

        geometry = {
            camera:
                self._geometry_fusion_record(
                    camera,
                    result,
                )
            for camera, result
            in geometry_results.items()
        }

        # --------------------------------------------------
        # 4. VLM sees the SAME frozen RGB frames.
        # --------------------------------------------------

        image_paths = (
            self._write_snapshot_images(
                snapshot
            )
        )

        try:
            vision = (
                self._vision_reasoner.analyze(
                    images=image_paths,
                    scene_context=(
                        scene_context or {}
                    ),
                    task=(
                        task
                        or (
                            f"Determine whether the "
                            f"target '{self.query}' is "
                            f"visually observable from "
                            f"each camera."
                        )
                    ),
                )
            )

        finally:
            # Files are only a transport format for the VLM.
            # The authoritative observation remains the frozen
            # in-memory snapshot.
            self._cleanup_snapshot_images(
                image_paths
            )

        # --------------------------------------------------
        # 5. Spatial + AI evidence fusion.
        # --------------------------------------------------

        fusion = (
            self._evidence_fusion.fuse(
                vision=vision,
                geometry=geometry,
            )
        )

        best_camera = fusion.get(
            "best_camera"
        )

        safe = bool(
            fusion.get(
                "safe_visual_evidence",
                False,
            )
        )

        scene = None

        if safe and best_camera is not None:
            selected = geometry_results.get(
                best_camera
            )

            if selected is not None:
                scene = selected.scene

        # --------------------------------------------------
        # Manipulation geometry selection.
        #
        # The evidence-fusion best camera is not necessarily
        # the camera with complete grasp geometry.
        #
        # Example:
        #   side camera -> strong visual/metric evidence
        #               -> no support_z / object height
        #
        # Manipulation requires a SceneObject with:
        #   position_robot + support_z + height
        #
        # Do not change the semantic best_camera decision.
        # Expose a separate manipulation scene instead.
        # --------------------------------------------------
        manipulation_scene = None
        manipulation_camera = None

        if safe:
            preferred = (
                best_camera,
                "main",
                "wrist",
                "side",
            )

            checked = set()

            for camera in preferred:
                if (
                    camera is None
                    or camera in checked
                ):
                    continue

                checked.add(camera)

                result = geometry_results.get(
                    camera
                )

                if (
                    result is None
                    or result.scene is None
                    or self.query
                    not in result.scene.objects
                ):
                    continue

                obj = result.scene.objects[
                    self.query
                ]

                if (
                    obj.position_robot is None
                    or obj.support_z is None
                    or obj.height is None
                ):
                    continue

                manipulation_scene = (
                    result.scene
                )
                manipulation_camera = (
                    camera
                )
                break

        # Defense in depth:
        # safe visual evidence still requires a SceneState.
        if scene is None:
            safe = False
            manipulation_scene = None
            manipulation_camera = None

        perception_summary = {
            "safe_visual_evidence": safe,
            "best_camera": (
                best_camera
                if safe
                else None
            ),
            "vision": vision,
            "fusion": fusion,
        }

        viewpoint_decision = (
            self._viewpoint_selector.select(
                perception_summary
            )
        )

        return {
            "safe_visual_evidence": safe,
            "scene": scene,
            "best_camera": (
                best_camera
                if safe
                else None
            ),

            "manipulation_scene": (
                manipulation_scene
                if safe
                else None
            ),
            "manipulation_camera": (
                manipulation_camera
                if safe
                else None
            ),

            "snapshot": snapshot,
            "inference": inference_results,
            "geometry_results": (
                geometry_results
            ),
            "geometry": geometry,
            "vision": vision,
            "fusion": fusion,

            "viewpoint_decision":
                viewpoint_decision,

            "reason": (
                "VERIFIED_SCENE_AVAILABLE"
                if safe
                else "NO_VERIFIED_SCENE"
            ),
        }


    def close(self):
        if self._closed:
            return

        self.stop_continuous_perception()

        if self._executor is not None:
            try:
                self._executor.shutdown(
                    timeout_sec=2.0
                )
            except Exception:
                pass

        if self._executor_thread is not None:
            try:
                self._executor_thread.join(
                    timeout=2.0
                )
            except Exception:
                pass

        for observer in (
            self.main_observer,
            self.side_observer,
            self.wrist_observer,
        ):
            try:
                observer.destroy_node()
            except Exception:
                pass

        self._executor = None
        self._executor_thread = None
        self._executor_started = False
        self._closed = True


    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ):
        self.close()
