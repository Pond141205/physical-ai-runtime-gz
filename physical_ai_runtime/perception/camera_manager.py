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
        if self._inference_manager is None:
            self._inference_manager = InferenceManager()

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

        # Defense in depth:
        # safe evidence without SceneState is not authorized.
        if scene is None:
            safe = False

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
