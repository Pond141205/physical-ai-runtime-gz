import time
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped

from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point

from object_detection.open_vocab_detector import OpenVocabularyDetector
from scene_state import SceneState, SceneObject
from robot_self_mask import RobotSelfMask
from threading import Lock


class RGBDSceneObserver(Node):

    def __init__(
        self,
        query="cube",
        robot_frame="panda_link0",
        robot_description_node="/panda/robot_state_publisher",
        self_mask_overlap_reject=0.35,
        detector=None,
        initialize_detector=True,
    ):
        super().__init__("rgbd_scene_observer")

        self.query = query
        self.robot_frame = robot_frame
        self.sensor_frame = None
        self.camera_frame = "runtime_rgbd_camera_optical_frame"

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

        # Diagnostic semantic state for callers.
        self.last_observation_status = None
        self.last_observation_metrics = {}

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
            "/runtime/rgbd/image",
            self.rgb_cb,
            10,
        )

        self.create_subscription(
            Image,
            "/runtime/rgbd/depth_image",
            self.depth_cb,
            10,
        )

        self.create_subscription(
            CameraInfo,
            "/runtime/rgbd/camera_info",
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

        # image/depth optical coordinates
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

        # Estimate support surface from a ring around the detected bbox.
        pad = 20

        rx1 = max(0, x1 - pad)
        ry1 = max(0, y1 - pad)
        rx2 = min(w, x2 + pad)
        ry2 = min(h, y2 + pad)

        ring_depth = self.depth[
            ry1:ry2,
            rx1:rx2
        ].copy()

        ring_mask = np.ones(
            ring_depth.shape,
            dtype=bool
        )

        inner_x1 = x1 - rx1
        inner_y1 = y1 - ry1
        inner_x2 = x2 - rx1
        inner_y2 = y2 - ry1

        ring_mask[
            inner_y1:inner_y2,
            inner_x1:inner_x2
        ] = False

        support_values = ring_depth[
            ring_mask
            & np.isfinite(ring_depth)
            & (ring_depth > 0.05)
            & (ring_depth < 3.0)
        ]

        support_z = None
        object_height = None

        if len(support_values) > 0:
            support_depth = float(
                np.median(support_values)
            )

            support_xo = (
                (u - self.cx)
                * support_depth
                / self.fx
            )

            support_yo = (
                (v - self.cy)
                * support_depth
                / self.fy
            )

            try:
                support_world = self._optical_to_world(
                    support_xo,
                    support_yo,
                    support_depth,
                    timestamp=timestamp,
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

        return world, z, support_z, object_height

    def observe_once(self, timeout=10.0):
        end_time = time.time() + timeout

        self.last_observation_status = None
        self.last_observation_metrics = {}

        while rclpy.ok() and time.time() < end_time:

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
                self.query,
            )

            if not detections:
                continue

            # Build current robot self-mask from live URDF + TF.
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

            # Validate every semantic candidate instead of blindly
            # trusting the detector's highest-confidence bbox.
            candidates = []

            image_h, image_w = self.rgb.shape[:2]

            total_detections = len(detections)
            self_mask_rejections = 0

            for detection in detections:
                x1, y1, x2, y2 = detection.bbox

                # Invalid / degenerate image geometry.
                if x2 <= x1 or y2 <= y1:
                    continue

                # Detections truncated by the image boundary are unreliable
                # for metric object geometry.
                touches_border = (
                    x1 <= 0
                    or y1 <= 0
                    or x2 >= image_w - 1
                    or y2 >= image_h - 1
                )

                result = self._bbox_to_world(
                    detection.bbox,
                    timestamp=self.observation_timestamp,
                )

                if result is None:
                    continue

                (
                    world_xyz,
                    depth,
                    support_z,
                    object_height,
                ) = result

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
                        "Rejected semantic candidate: "
                        f"foreground_robot_overlap="
                        f"{self_overlap:.3f} "
                        f"object_depth={depth:.3f} "
                        f"bbox={detection.bbox}"
                    )
                    continue

                (
                    candidate_world,
                    candidate_depth,
                    candidate_support_z,
                    candidate_height,
                ) = result

                # Metric depth/support validity.
                if not np.all(
                    np.isfinite(candidate_world)
                ):
                    continue

                if (
                    not np.isfinite(candidate_depth)
                    or candidate_depth <= 0.0
                ):
                    continue

                if (
                    candidate_support_z is None
                    or candidate_height is None
                ):
                    continue

                if (
                    not np.isfinite(candidate_support_z)
                    or not np.isfinite(candidate_height)
                    or candidate_height <= 0.0
                ):
                    continue

                # Border candidates are retained only as lower-priority
                # hypotheses; they must never win over a complete detection.
                geometry_penalty = (
                    1.0 if touches_border else 0.0
                )

                candidates.append((
                    geometry_penalty,
                    -float(detection.confidence),
                    detection,
                    candidate_world,
                    candidate_depth,
                    candidate_support_z,
                    candidate_height,
                ))

            if not candidates:
                if (
                    total_detections > 0
                    and self_mask_rejections
                    == total_detections
                ):
                    self.last_observation_status = (
                        "SELF_OCCLUDED"
                    )

                    self.last_observation_metrics = {
                        "detections":
                            int(total_detections),
                        "self_mask_rejections":
                            int(self_mask_rejections),
                        "self_mask_overlap_reject":
                            float(
                                self.self_mask_overlap_reject
                            ),
                    }

                continue

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

            self.last_observation_status = (
                "OBJECT_VISIBLE"
            )

            self.last_observation_metrics = {
                "detections":
                    int(total_detections),
                "self_mask_rejections":
                    int(self_mask_rejections),
            }

            obj = SceneObject(
                object_id=self.query,
                position_world=world_xyz,
                position_robot=robot_xyz,
                depth=depth,
                height=object_height,
                support_z=support_z,
            )

            return SceneState(
                objects={
                    self.query: obj
                }
            )

        if self.last_observation_status is None:
            self.last_observation_status = (
                "OBJECT_NOT_DETECTED"
            )

        return None


def main():
    rclpy.init()

    observer = RGBDSceneObserver(
        query="cube"
    )

    scene = observer.observe_once()

    print("SCENE:")
    print(scene)

    if scene is not None:
        cube = scene.objects["cube"]

        print("CUBE WORLD:")
        print(cube.position_world)

        print("CUBE ROBOT:")
        print(cube.position_robot)

    observer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
