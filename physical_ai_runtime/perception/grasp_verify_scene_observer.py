import time
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped

from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point

from physical_ai_runtime.perception.object_detection.open_vocab_detector import OpenVocabularyDetector
from physical_ai_runtime.perception.scene_state import SceneState, SceneObject
from physical_ai_runtime.perception.robot_self_mask import RobotSelfMask
from threading import Lock


class GraspVerifySceneObserver(Node):

    def __init__(
        self,
        query="cube",
        robot_frame="panda_link0",
        reference_world=None,
        reference_size=None,
        workspace=None,
        robot_description_node="/panda/robot_state_publisher",
        self_mask_overlap_reject=0.35,
        detector=None,
        initialize_detector=True,
    ):
        super().__init__("grasp_verify_scene_observer")

        self.query = query
        self.robot_frame = robot_frame
        self.sensor_frame = None
        self.camera_frame = "grasp_verify_camera_optical_frame"

        self.reference_world = (
            None
            if reference_world is None
            else np.asarray(reference_world, dtype=float)
        )

        self.reference_size = (
            None
            if reference_size is None
            else np.asarray(reference_size, dtype=float)
        )

        self.workspace = workspace

        if detector is not None:
            self.detector = detector

        elif initialize_detector:
            self.detector = OpenVocabularyDetector(
                box_threshold=0.20,
                text_threshold=0.20,
            )

        else:
            self.detector = None

        self.rgb = None
        self.depth = None

        self.rgb_timestamp = None
        self.depth_timestamp = None
        self.observation_timestamp = None
        self.rgb_depth_sync_tolerance_s = 0.050

        # Protect RGB, depth, intrinsics and timestamps as one
        # coherent sensor state for concurrent real-world streams.
        self.sensor_lock = Lock()

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        self.self_mask_overlap_reject = float(
            self_mask_overlap_reject
        )

        self.robot_self_mask = RobotSelfMask(
            node=self,
            tf_buffer=self.tf_buffer,
            camera_frame=self.camera_frame,
            robot_description_node=robot_description_node,
        )

        self.create_subscription(
            Image,
            "/runtime/grasp_verify/image",
            self.rgb_cb,
            10,
        )

        self.create_subscription(
            Image,
            "/runtime/grasp_verify/depth_image",
            self.depth_cb,
            10,
        )

        self.create_subscription(
            CameraInfo,
            "/runtime/grasp_verify/camera_info",
            self.info_cb,
            10,
        )

    @staticmethod
    def _stamp_to_seconds(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _update_observation_timestamp(self):
        if self.rgb_timestamp is None or self.depth_timestamp is None:
            self.observation_timestamp = None
            return False

        delta = abs(self.rgb_timestamp - self.depth_timestamp)

        if delta > self.rgb_depth_sync_tolerance_s:
            self.observation_timestamp = None
            return False

        self.observation_timestamp = 0.5 * (
            self.rgb_timestamp + self.depth_timestamp
        )

        return True

    def rgb_cb(self, msg):
        timestamp = self._stamp_to_seconds(
            msg.header.stamp
        )

        arr = np.frombuffer(
            msg.data,
            dtype=np.uint8,
        )

        enc = msg.encoding.lower()

        if enc == "rgb8":
            image = arr.reshape(
                msg.height,
                msg.width,
                3,
            ).copy()

        elif enc == "bgr8":
            image = arr.reshape(
                msg.height,
                msg.width,
                3,
            )[:, :, ::-1].copy()

        else:
            return

        # Publish image + timestamp atomically.
        with self.sensor_lock:
            self.rgb = image
            self.rgb_timestamp = timestamp
            self._update_observation_timestamp()


    def depth_cb(self, msg):
        timestamp = self._stamp_to_seconds(
            msg.header.stamp
        )

        depth = np.frombuffer(
            msg.data,
            dtype=np.float32,
        ).reshape(
            msg.height,
            msg.width,
        ).copy()

        # Publish depth + timestamp atomically.
        with self.sensor_lock:
            self.depth = depth
            self.depth_timestamp = timestamp
            self._update_observation_timestamp()


    def info_cb(self, msg):
        with self.sensor_lock:
            self.fx = float(msg.k[0])
            self.fy = float(msg.k[4])
            self.cx = float(msg.k[2])
            self.cy = float(msg.k[5])


    def snapshot_sensor_state(self):
        """
        Return one immutable copy of the latest RGB-D sensor state.

        This method performs no inference and no robot motion.
        """
        with self.sensor_lock:
            rgb = (
                None
                if self.rgb is None
                else self.rgb.copy()
            )

            depth = (
                None
                if self.depth is None
                else self.depth.copy()
            )

            return {
                "rgb": rgb,
                "depth": depth,

                "rgb_timestamp": self.rgb_timestamp,
                "depth_timestamp": self.depth_timestamp,
                "observation_timestamp": (
                    self.observation_timestamp
                ),

                "fx": self.fx,
                "fy": self.fy,
                "cx": self.cx,
                "cy": self.cy,

                "sensor_frame": getattr(
                    self,
                    "sensor_frame",
                    None,
                ),

                "camera_frame": self.camera_frame,
            }


    def _tf_ready(self):
        """
        Fail closed until this observer's TF buffer has received
        the static/world transforms required for geometry.
        """
        try:
            self.tf_buffer.lookup_transform(
                "world",
                self.camera_frame,
                rclpy.time.Time(),
            )

            self.tf_buffer.lookup_transform(
                self.camera_frame,
                self.robot_frame,
                rclpy.time.Time(),
            )

            return True

        except Exception:
            return False


    def _ready(self):
        return (
            self.rgb is not None
            and self.depth is not None
            and self.fx is not None
        )

    def _quat_to_matrix(self, q):
        x, y, z, w = q

        return np.array([
            [
                1 - 2*(y*y + z*z),
                2*(x*y - z*w),
                2*(x*z + y*w)
            ],
            [
                2*(x*y + z*w),
                1 - 2*(x*x + z*z),
                2*(y*z - x*w)
            ],
            [
                2*(x*z - y*w),
                2*(y*z + x*w),
                1 - 2*(x*x + y*y)
            ]
        ], dtype=float)

    def _optical_to_world(
        self,
        x,
        y,
        z,
        timestamp=None,
    ):
        point = PointStamped()
        point.header.frame_id = self.camera_frame
        point.point.x = float(x)
        point.point.y = float(y)
        point.point.z = float(z)

        if timestamp is None:
            tf_time = Time()
        else:
            tf_time = Time(
                nanoseconds=int(
                    float(timestamp) * 1_000_000_000
                )
            )

        transform = self.tf_buffer.lookup_transform(
            "world",
            self.camera_frame,
            tf_time,
        )

        transformed = do_transform_point(
            point,
            transform
        )

        return np.array([
            transformed.point.x,
            transformed.point.y,
            transformed.point.z,
        ], dtype=float)

    def _bbox_to_world(
        self,
        bbox,
        timestamp=None,
    ):
        x1, y1, x2, y2 = bbox

        h, w = self.depth.shape

        x1 = max(0, min(w - 1, x1))
        x2 = max(1, min(w, x2))

        y1 = max(0, min(h - 1, y1))
        y2 = max(1, min(h, y2))

        crop = self.depth[
            y1:y2,
            x1:x2
        ]

        valid = crop[
            np.isfinite(crop)
            & (crop > 0.05)
            & (crop < 3.0)
        ]

        if len(valid) == 0:
            return None

        z = float(np.median(valid))

        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0

        xo = (u - self.cx) * z / self.fx
        yo = (v - self.cy) * z / self.fy
        zo = z

        try:
            world = self._optical_to_world(
                xo,
                yo,
                zo,
                timestamp=timestamp,
            )
        except Exception:
            return None

        return world, z

    def observe_once(self, timeout=10.0):
        end = time.time() + timeout

        self.last_observation_status = None
        self.last_observation_metrics = {}

        total_detections_seen = 0
        self_mask_rejections = 0

        while rclpy.ok() and time.time() < end:

            rclpy.spin_once(
                self,
                timeout_sec=0.1
            )

            if not self._ready():
                continue

            if not self._tf_ready():
                continue

            detections = self.detector.detect(
                self.rgb,
                self.query
            )

            if not detections:
                continue

            total_detections_seen += len(detections)

            try:
                robot_depth_map = (
                    self.robot_self_mask.build_depth_map(
                        image_shape=self.rgb.shape,
                        fx=self.fx,
                        fy=self.fy,
                        cx=self.cx,
                        cy=self.cy,
                        timestamp=self.observation_timestamp,
                    )
                )
            except Exception as e:
                self.get_logger().warning(
                    f"SELF_DEPTH_UNAVAILABLE: {e}"
                )
                continue

            candidates = []

            for detection in detections:

                # First obtain metric target depth.
                #
                # Occlusion cannot be decided correctly from
                # 2D silhouette overlap alone.
                result = self._bbox_to_world(
                    detection.bbox,
                    timestamp=self.observation_timestamp,
                )

                if result is None:
                    continue

                world_xyz, depth = result

                self_overlap = (
                    self.robot_self_mask
                    .depth_aware_bbox_overlap_ratio(
                        robot_depth_map,
                        detection.bbox,
                        object_depth=depth,
                        depth_margin=0.015,
                    )
                )

                if (
                    self_overlap
                    >= self.self_mask_overlap_reject
                ):
                    self_mask_rejections += 1

                    self.get_logger().info(
                        "Rejected verify candidate: "
                        f"foreground_robot_overlap="
                        f"{self_overlap:.3f} "
                        f"object_depth={depth:.3f} "
                        f"bbox={detection.bbox}"
                    )
                    continue

                self.get_logger().info(
                    "Accepted depth-aware candidate: "
                    f"foreground_robot_overlap="
                    f"{self_overlap:.3f} "
                    f"object_depth={depth:.3f} "
                    f"bbox={detection.bbox}"
                )

                # Physical workspace validation.
                if self.workspace is not None:

                    if not (
                        self.workspace["x_min"]
                        <= world_xyz[0]
                        <= self.workspace["x_max"]
                    ):
                        continue

                    if not (
                        self.workspace["y_min"]
                        <= world_xyz[1]
                        <= self.workspace["y_max"]
                    ):
                        continue

                    # Object must not be below the support surface.
                    if (
                        world_xyz[2]
                        < self.workspace["z"] - 0.005
                    ):
                        continue

                x1, y1, x2, y2 = detection.bbox

                pixel_w = max(1, x2 - x1)
                pixel_h = max(1, y2 - y1)

                metric_w = (
                    pixel_w * depth / self.fx
                )

                metric_h = (
                    pixel_h * depth / self.fy
                )

                # Reject detections spanning an implausibly large
                # fraction of the workspace / scene.
                if metric_w > 0.20 or metric_h > 0.20:
                    continue

                # If we already observed this object from this camera,
                # reject candidates whose apparent size changed too much.
                if self.reference_size is not None:

                    ref_w = float(self.reference_size[0])
                    ref_h = float(self.reference_size[1])

                    if ref_w > 0.0 and ref_h > 0.0:

                        width_ratio = metric_w / ref_w
                        height_ratio = metric_h / ref_h

                        if not (
                            0.50 <= width_ratio <= 1.75
                            and
                            0.50 <= height_ratio <= 1.75
                        ):
                            continue

                if self.reference_world is not None:
                    distance = float(
                        np.linalg.norm(
                            world_xyz
                            - self.reference_world
                        )
                    )
                else:
                    distance = 0.0

                # Prefer spatial consistency first,
                # detector confidence second.
                score = (
                    -distance
                    + 0.05 * float(
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
                    )
                )

            if not candidates:
                continue

            candidates.sort(
                key=lambda x: x[0],
                reverse=True
            )

            (
                _,
                detection,
                world_xyz,
                depth,
                metric_w,
                metric_h,
            ) = candidates[0]

            self.get_logger().info(
                "Selected %s world=%s size≈[%.3f, %.3f] m"
                % (
                    detection.bbox,
                    np.round(world_xyz, 4),
                    metric_w,
                    metric_h,
                )
            )

            p = PointStamped()
            p.header.frame_id = "world"

            p.point.x = float(world_xyz[0])
            p.point.y = float(world_xyz[1])
            p.point.z = float(world_xyz[2])

            try:
                tf = self.tf_buffer.lookup_transform(
                    self.robot_frame,
                    "world",
                    rclpy.time.Time()
                )

                rp = do_transform_point(
                    p,
                    tf
                )

            except Exception as e:
                self.get_logger().warning(
                    f"TF world -> {self.robot_frame} failed: {e}"
                )
                continue

            robot_xyz = np.array([
                rp.point.x,
                rp.point.y,
                rp.point.z
            ], dtype=float)

            obj = SceneObject(
                object_id=self.query,
                position_world=world_xyz,
                position_robot=robot_xyz,
                depth=depth,
                size_xyz=np.array([
                    metric_w,
                    metric_h,
                    0.0
                ], dtype=float),
            )

            self.last_observation_status = (
                "OBJECT_VISIBLE"
            )

            self.last_observation_metrics = {
                "detections": total_detections_seen,
                "self_mask_rejections": self_mask_rejections,
                "depth": float(depth),
                "world_xyz": [
                    float(world_xyz[0]),
                    float(world_xyz[1]),
                    float(world_xyz[2]),
                ],
            }

            return SceneState(
                objects={
                    self.query: obj
                }
            )

        if self.last_observation_status is None:
            if (
                total_detections_seen > 0
                and self_mask_rejections
                == total_detections_seen
            ):
                self.last_observation_status = (
                    "SELF_OCCLUDED"
                )
            else:
                self.last_observation_status = (
                    "OBJECT_NOT_DETECTED"
                )

            self.last_observation_metrics = {
                "detections": total_detections_seen,
                "self_mask_rejections": self_mask_rejections,
            }

        return None


def main():
    rclpy.init()

    observer = GraspVerifySceneObserver(
        query="cube"
    )

    scene = observer.observe_once(
        timeout=10.0
    )

    print("VERIFY SCENE:")
    print(scene)

    if scene is not None:
        cube = scene.objects["cube"]

        print("VERIFY CUBE WORLD:")
        print(cube.position_world)

        print("VERIFY CUBE ROBOT:")
        print(cube.position_robot)

    observer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
