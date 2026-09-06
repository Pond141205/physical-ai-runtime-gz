from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from geometry_msgs.msg import PointStamped
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_point

from physical_ai_runtime.perception.scene_state import SceneObject, SceneState


@dataclass
class GeometryResult:
    camera: str
    status: str
    scene: Optional[SceneState]
    metrics: Dict[str, Any]


class SnapshotGeometryProcessor:
    """
    Deterministic metric geometry over immutable RGB-D snapshots.

    Sensor data always comes from CameraSnapshot.
    Observer objects are used only for:
      - TF buffer
      - RobotSelfMask / robot collision geometry
      - camera / robot frame configuration
      - camera-specific validation parameters

    No ROS spinning.
    No detector inference.
    No robot motion.
    """

    def __init__(
        self,
        query="cube",
        depth_min_m=0.05,
        depth_max_m=3.0,
        self_depth_margin_m=0.015,
    ):
        self.query = query
        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)
        self.self_depth_margin_m = float(
            self_depth_margin_m
        )

    @staticmethod
    def _tf_time(timestamp):
        return Time(
            nanoseconds=int(
                float(timestamp) * 1_000_000_000
            )
        )

    def _optical_to_world(
        self,
        observer,
        snapshot,
        x,
        y,
        z,
    ):
        point = PointStamped()
        point.header.frame_id = (
            snapshot.camera_frame
        )

        point.point.x = float(x)
        point.point.y = float(y)
        point.point.z = float(z)

        transform = (
            observer.tf_buffer.lookup_transform(
                "world",
                snapshot.camera_frame,
                self._tf_time(
                    snapshot.observation_timestamp
                ),
            )
        )

        transformed = do_transform_point(
            point,
            transform,
        )

        return np.array(
            [
                transformed.point.x,
                transformed.point.y,
                transformed.point.z,
            ],
            dtype=float,
        )

    def _world_to_robot(
        self,
        observer,
        snapshot,
        world_xyz,
    ):
        point = PointStamped()
        point.header.frame_id = "world"

        point.point.x = float(world_xyz[0])
        point.point.y = float(world_xyz[1])
        point.point.z = float(world_xyz[2])

        transform = (
            observer.tf_buffer.lookup_transform(
                observer.robot_frame,
                "world",
                self._tf_time(
                    snapshot.observation_timestamp
                ),
            )
        )

        transformed = do_transform_point(
            point,
            transform,
        )

        return np.array(
            [
                transformed.point.x,
                transformed.point.y,
                transformed.point.z,
            ],
            dtype=float,
        )

    def _bbox_depth(
        self,
        snapshot,
        bbox,
    ):
        x1, y1, x2, y2 = [
            int(round(v))
            for v in bbox
        ]

        h, w = snapshot.depth.shape

        x1 = max(0, min(w - 1, x1))
        x2 = max(1, min(w, x2))

        y1 = max(0, min(h - 1, y1))
        y2 = max(1, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = snapshot.depth[
            y1:y2,
            x1:x2,
        ]

        valid = crop[
            np.isfinite(crop)
            & (crop > self.depth_min_m)
            & (crop < self.depth_max_m)
        ]

        if len(valid) == 0:
            return None

        depth = float(
            np.median(valid)
        )

        return (
            x1,
            y1,
            x2,
            y2,
            depth,
        )

    def _bbox_to_world_basic(
        self,
        observer,
        snapshot,
        bbox,
    ):
        result = self._bbox_depth(
            snapshot,
            bbox,
        )

        if result is None:
            return None

        x1, y1, x2, y2, depth = result

        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0

        xo = (
            (u - snapshot.cx)
            * depth
            / snapshot.fx
        )

        yo = (
            (v - snapshot.cy)
            * depth
            / snapshot.fy
        )

        try:
            world = self._optical_to_world(
                observer,
                snapshot,
                xo,
                yo,
                depth,
            )
        except Exception:
            return None

        return world, depth

    def _bbox_to_world_support(
        self,
        observer,
        snapshot,
        bbox,
    ):
        basic = self._bbox_to_world_basic(
            observer,
            snapshot,
            bbox,
        )

        if basic is None:
            return None

        world, depth = basic

        x1, y1, x2, y2 = [
            int(round(v))
            for v in bbox
        ]

        h, w = snapshot.depth.shape

        x1 = max(0, min(w - 1, x1))
        x2 = max(1, min(w, x2))

        y1 = max(0, min(h - 1, y1))
        y2 = max(1, min(h, y2))

        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0

        pad = 20

        rx1 = max(0, x1 - pad)
        ry1 = max(0, y1 - pad)
        rx2 = min(w, x2 + pad)
        ry2 = min(h, y2 + pad)

        ring_depth = snapshot.depth[
            ry1:ry2,
            rx1:rx2,
        ].copy()

        ring_mask = np.ones(
            ring_depth.shape,
            dtype=bool,
        )

        inner_x1 = x1 - rx1
        inner_y1 = y1 - ry1
        inner_x2 = x2 - rx1
        inner_y2 = y2 - ry1

        ring_mask[
            inner_y1:inner_y2,
            inner_x1:inner_x2,
        ] = False

        support_values = ring_depth[
            ring_mask
            & np.isfinite(ring_depth)
            & (ring_depth > self.depth_min_m)
            & (ring_depth < self.depth_max_m)
        ]

        support_z = None
        object_height = None

        if len(support_values) > 0:
            support_depth = float(
                np.median(support_values)
            )

            support_xo = (
                (u - snapshot.cx)
                * support_depth
                / snapshot.fx
            )

            support_yo = (
                (v - snapshot.cy)
                * support_depth
                / snapshot.fy
            )

            try:
                support_world = (
                    self._optical_to_world(
                        observer,
                        snapshot,
                        support_xo,
                        support_yo,
                        support_depth,
                    )
                )
            except Exception:
                support_world = None

            if support_world is not None:
                support_z = float(
                    support_world[2]
                )

                object_height = float(
                    world[2] - support_z
                )

                if object_height <= 0.0:
                    object_height = None

        return (
            world,
            depth,
            support_z,
            object_height,
        )

    def _self_depth_map(
        self,
        observer,
        snapshot,
    ):
        return (
            observer.robot_self_mask
            .build_depth_map(
                image_shape=snapshot.rgb.shape,
                fx=snapshot.fx,
                fy=snapshot.fy,
                cx=snapshot.cx,
                cy=snapshot.cy,
                timestamp=(
                    snapshot.observation_timestamp
                ),
            )
        )

    def _process_main_or_wrist(
        self,
        camera,
        observer,
        snapshot,
        detections,
    ):
        total = len(detections)

        if total == 0:
            return GeometryResult(
                camera=camera,
                status="OBJECT_NOT_DETECTED",
                scene=None,
                metrics={
                    "detections": 0,
                    "self_mask_rejections": 0,
                    "timestamp":
                        snapshot.observation_timestamp,
                },
            )

        try:
            robot_depth_map = (
                self._self_depth_map(
                    observer,
                    snapshot,
                )
            )
        except Exception as e:
            return GeometryResult(
                camera=camera,
                status="GEOMETRY_UNVERIFIED",
                scene=None,
                metrics={
                    "detections": total,
                    "error":
                        f"SELF_DEPTH_UNAVAILABLE: {e}",
                    "timestamp":
                        snapshot.observation_timestamp,
                },
            )

        candidates = []
        self_rejections = 0

        image_h, image_w = (
            snapshot.rgb.shape[:2]
        )

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox

            if x2 <= x1 or y2 <= y1:
                continue

            touches_border = (
                x1 <= 0
                or y1 <= 0
                or x2 >= image_w - 1
                or y2 >= image_h - 1
            )

            result = (
                self._bbox_to_world_support(
                    observer,
                    snapshot,
                    detection.bbox,
                )
            )

            if result is None:
                continue

            (
                world_xyz,
                depth,
                support_z,
                object_height,
            ) = result

            overlap = (
                observer.robot_self_mask
                .depth_aware_bbox_overlap_ratio(
                    robot_depth_map,
                    detection.bbox,
                    object_depth=depth,
                    depth_margin=(
                        self.self_depth_margin_m
                    ),
                )
            )

            if (
                overlap
                >= observer.self_mask_overlap_reject
            ):
                self_rejections += 1
                continue

            if not np.all(
                np.isfinite(world_xyz)
            ):
                continue

            if (
                not np.isfinite(depth)
                or depth <= 0.0
            ):
                continue

            if (
                support_z is None
                or object_height is None
            ):
                continue

            if (
                not np.isfinite(support_z)
                or not np.isfinite(object_height)
                or object_height <= 0.0
            ):
                continue

            geometry_penalty = (
                1.0
                if touches_border
                else 0.0
            )

            candidates.append(
                (
                    geometry_penalty,
                    -float(
                        detection.confidence
                    ),
                    detection,
                    world_xyz,
                    depth,
                    support_z,
                    object_height,
                )
            )

        if not candidates:
            status = (
                "SELF_OCCLUDED"
                if (
                    total > 0
                    and self_rejections == total
                )
                else "OBJECT_NOT_DETECTED"
            )

            return GeometryResult(
                camera=camera,
                status=status,
                scene=None,
                metrics={
                    "detections": total,
                    "self_mask_rejections":
                        self_rejections,
                    "timestamp":
                        snapshot.observation_timestamp,
                },
            )

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
            )
        )

        (
            _,
            _,
            detection,
            world_xyz,
            depth,
            support_z,
            object_height,
        ) = candidates[0]

        try:
            robot_xyz = self._world_to_robot(
                observer,
                snapshot,
                world_xyz,
            )
        except Exception as e:
            return GeometryResult(
                camera=camera,
                status="GEOMETRY_UNVERIFIED",
                scene=None,
                metrics={
                    "detections": total,
                    "error":
                        f"WORLD_TO_ROBOT_TF_FAILED: {e}",
                    "timestamp":
                        snapshot.observation_timestamp,
                },
            )

        obj = SceneObject(
            object_id=self.query,
            position_world=world_xyz,
            position_robot=robot_xyz,
            depth=depth,
            height=object_height,
            support_z=support_z,
        )

        return GeometryResult(
            camera=camera,
            status="OBJECT_VISIBLE",
            scene=SceneState(
                objects={
                    self.query: obj
                }
            ),
            metrics={
                "detections": total,
                "self_mask_rejections":
                    self_rejections,
                "depth": float(depth),
                "world_xyz": (
                    world_xyz.tolist()
                ),
                "timestamp":
                    snapshot.observation_timestamp,
                "selected_bbox":
                    tuple(detection.bbox),
            },
        )

    def _process_side(
        self,
        observer,
        snapshot,
        detections,
        reference_world=None,
    ):
        total = len(detections)

        if total == 0:
            return GeometryResult(
                camera="side",
                status="OBJECT_NOT_DETECTED",
                scene=None,
                metrics={
                    "detections": 0,
                    "self_mask_rejections": 0,
                    "timestamp":
                        snapshot.observation_timestamp,
                },
            )

        try:
            robot_depth_map = (
                self._self_depth_map(
                    observer,
                    snapshot,
                )
            )
        except Exception as e:
            return GeometryResult(
                camera="side",
                status="GEOMETRY_UNVERIFIED",
                scene=None,
                metrics={
                    "detections": total,
                    "error":
                        f"SELF_DEPTH_UNAVAILABLE: {e}",
                },
            )

        candidates = []
        self_rejections = 0

        for detection in detections:
            result = (
                self._bbox_to_world_basic(
                    observer,
                    snapshot,
                    detection.bbox,
                )
            )

            if result is None:
                continue

            world_xyz, depth = result

            overlap = (
                observer.robot_self_mask
                .depth_aware_bbox_overlap_ratio(
                    robot_depth_map,
                    detection.bbox,
                    object_depth=depth,
                    depth_margin=(
                        self.self_depth_margin_m
                    ),
                )
            )

            if (
                overlap
                >= observer.self_mask_overlap_reject
            ):
                self_rejections += 1
                continue

            workspace = observer.workspace

            if workspace is not None:
                if not (
                    workspace["x_min"]
                    <= world_xyz[0]
                    <= workspace["x_max"]
                ):
                    continue

                if not (
                    workspace["y_min"]
                    <= world_xyz[1]
                    <= workspace["y_max"]
                ):
                    continue

                if (
                    world_xyz[2]
                    < workspace["z"] - 0.005
                ):
                    continue

            x1, y1, x2, y2 = detection.bbox

            pixel_w = max(1, x2 - x1)
            pixel_h = max(1, y2 - y1)

            metric_w = (
                pixel_w
                * depth
                / snapshot.fx
            )

            metric_h = (
                pixel_h
                * depth
                / snapshot.fy
            )

            if (
                metric_w > 0.20
                or metric_h > 0.20
            ):
                continue

            if observer.reference_size is not None:
                ref_w = float(
                    observer.reference_size[0]
                )

                ref_h = float(
                    observer.reference_size[1]
                )

                if ref_w > 0.0 and ref_h > 0.0:
                    width_ratio = (
                        metric_w / ref_w
                    )

                    height_ratio = (
                        metric_h / ref_h
                    )

                    if not (
                        0.50 <= width_ratio <= 1.75
                        and
                        0.50 <= height_ratio <= 1.75
                    ):
                        continue

            candidate_reference = (
                reference_world
                if reference_world is not None
                else observer.reference_world
            )

            if candidate_reference is not None:
                distance = float(
                    np.linalg.norm(
                        world_xyz - candidate_reference
                    )
                )
            else:
                distance = 0.0

            score = (
                -distance
                + 0.05
                * float(
                    detection.confidence
                )
            )

            candidates.append(
                (
                    score,
                    detection,
                    world_xyz,
                    depth,
                    metric_w,
                    metric_h,
                    distance,
                )
            )

        if not candidates:
            status = (
                "SELF_OCCLUDED"
                if (
                    total > 0
                    and self_rejections == total
                )
                else "OBJECT_NOT_DETECTED"
            )

            return GeometryResult(
                camera="side",
                status=status,
                scene=None,
                metrics={
                    "detections": total,
                    "self_mask_rejections":
                        self_rejections,
                    "timestamp":
                        snapshot.observation_timestamp,
                },
            )

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        (
            _,
            detection,
            world_xyz,
            depth,
            metric_w,
            metric_h,
            reference_distance,
        ) = candidates[0]

        try:
            robot_xyz = self._world_to_robot(
                observer,
                snapshot,
                world_xyz,
            )
        except Exception as e:
            return GeometryResult(
                camera="side",
                status="GEOMETRY_UNVERIFIED",
                scene=None,
                metrics={
                    "detections": total,
                    "error":
                        f"WORLD_TO_ROBOT_TF_FAILED: {e}",
                    "timestamp":
                        snapshot.observation_timestamp,
                },
            )

        obj = SceneObject(
            object_id=self.query,
            position_world=world_xyz,
            position_robot=robot_xyz,
            depth=depth,
            size_xyz=np.array(
                [
                    metric_w,
                    metric_h,
                    0.0,
                ],
                dtype=float,
            ),
        )

        return GeometryResult(
            camera="side",
            status="OBJECT_VISIBLE",
            scene=SceneState(
                objects={
                    self.query: obj
                }
            ),
            metrics={
                "detections": total,
                "self_mask_rejections":
                    self_rejections,
                "depth": float(depth),
                "world_xyz":
                    world_xyz.tolist(),
                "metric_size": [
                    float(metric_w),
                    float(metric_h),
                ],
                "reference_distance_m": (
                    None
                    if reference_world is None
                    and observer.reference_world is None
                    else float(reference_distance)
                ),
                "timestamp":
                    snapshot.observation_timestamp,
                "selected_bbox":
                    tuple(detection.bbox),
            },
        )

    def process_camera(
        self,
        camera,
        observer,
        snapshot,
        inference_result,
    ):
        detections = (
            inference_result.detections
        )

        if camera == "side":
            return self._process_side(
                observer,
                snapshot,
                detections,
            )

        if camera in (
            "main",
            "wrist",
        ):
            return (
                self._process_main_or_wrist(
                    camera,
                    observer,
                    snapshot,
                    detections,
                )
            )

        return GeometryResult(
            camera=camera,
            status="GEOMETRY_UNVERIFIED",
            scene=None,
            metrics={
                "error":
                    "UNKNOWN_CAMERA",
            },
        )

    def process_multiview(
        self,
        manager,
        snapshot,
        inference_results,
    ):
        observers = {
            "main":
                manager.main_observer,
            "side":
                manager.side_observer,
            "wrist":
                manager.wrist_observer,
        }

        results = {}
        main_reference_world = None

        for camera in (
            "main",
            "side",
            "wrist",
        ):
            camera_snapshot = snapshot.get(
                camera
            )

            inference_result = (
                inference_results.get(
                    camera
                )
            )

            if (
                camera_snapshot is None
                or inference_result is None
            ):
                results[camera] = (
                    GeometryResult(
                        camera=camera,
                        status="GEOMETRY_UNVERIFIED",
                        scene=None,
                        metrics={
                            "error":
                                "MISSING_SNAPSHOT_OR_INFERENCE",
                        },
                    )
                )
                continue

            if camera == "side":
                results[camera] = self._process_side(
                    observers[camera],
                    camera_snapshot,
                    inference_result.detections,
                    reference_world=main_reference_world,
                )
            else:
                results[camera] = self.process_camera(
                    camera,
                    observers[camera],
                    camera_snapshot,
                    inference_result,
                )

            if camera == "main":
                main_scene = results[camera].scene
                main_object = (
                    None
                    if main_scene is None
                    else main_scene.objects.get(self.query)
                )

                if main_object is not None:
                    main_reference_world = (
                        main_object.position_world
                    )

        return results
