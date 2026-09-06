import math
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import trimesh

from rclpy.parameter_client import AsyncParameterClient
from rclpy.time import Time


class RobotSelfMask:
    """
    Conservative image-space mask generated from the robot's live URDF
    collision meshes and current TF transforms.

    No robot link names or robot dimensions are hardcoded.
    """

    def __init__(
        self,
        node,
        tf_buffer,
        camera_frame,
        robot_description_node,
    ):
        self.node = node
        self.tf_buffer = tf_buffer
        self.camera_frame = camera_frame
        self.robot_description_node = robot_description_node

        self._parameter_client = AsyncParameterClient(
            node,
            robot_description_node,
        )

        self._collision_meshes = None

    @staticmethod
    def _rpy_matrix(roll, pitch, yaw):
        cr = math.cos(roll)
        sr = math.sin(roll)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)

        return np.array([
            [
                cy * cp,
                cy * sp * sr - sy * cr,
                cy * sp * cr + sy * sr,
            ],
            [
                sy * cp,
                sy * sp * sr + cy * cr,
                sy * sp * cr - cy * sr,
            ],
            [
                -sp,
                cp * sr,
                cp * cr,
            ],
        ], dtype=float)

    @staticmethod
    def _quaternion_matrix(x, y, z, w):
        xx = x * x
        yy = y * y
        zz = z * z

        xy = x * y
        xz = x * z
        yz = y * z

        wx = w * x
        wy = w * y
        wz = w * z

        return np.array([
            [
                1.0 - 2.0 * (yy + zz),
                2.0 * (xy - wz),
                2.0 * (xz + wy),
            ],
            [
                2.0 * (xy + wz),
                1.0 - 2.0 * (xx + zz),
                2.0 * (yz - wx),
            ],
            [
                2.0 * (xz - wy),
                2.0 * (yz + wx),
                1.0 - 2.0 * (xx + yy),
            ],
        ], dtype=float)

    @staticmethod
    def _parse_vector(text, default):
        if text is None:
            return np.asarray(default, dtype=float)

        values = [
            float(v)
            for v in text.strip().split()
        ]

        if len(values) != 3:
            raise ValueError(
                f"Expected 3-vector, got: {text}"
            )

        return np.asarray(values, dtype=float)

    @staticmethod
    def _resolve_mesh_uri(uri):
        parsed = urlparse(uri)

        if parsed.scheme == "file":
            return Path(parsed.path)

        if parsed.scheme == "":
            return Path(uri)

        raise ValueError(
            f"Unsupported mesh URI: {uri}"
        )

    def _load_robot_description(self, timeout=3.0):
        if not self._parameter_client.wait_for_services(
            timeout_sec=timeout
        ):
            raise RuntimeError(
                "ROBOT_DESCRIPTION_SERVICE_UNAVAILABLE: "
                f"{self.robot_description_node}"
            )

        future = self._parameter_client.get_parameters([
            "robot_description"
        ])

        import rclpy

        rclpy.spin_until_future_complete(
            self.node,
            future,
            timeout_sec=timeout,
        )

        if not future.done():
            raise RuntimeError(
                "ROBOT_DESCRIPTION_TIMEOUT"
            )

        result = future.result()

        if (
            result is None
            or not result.values
            or not result.values[0].string_value
        ):
            raise RuntimeError(
                "ROBOT_DESCRIPTION_EMPTY"
            )

        return result.values[0].string_value

    def _load_collision_meshes(self):
        urdf_xml = self._load_robot_description()
        root = ET.fromstring(urdf_xml)

        collision_meshes = []

        for link in root.findall("link"):
            link_name = link.get("name")

            if not link_name:
                continue

            for collision in link.findall("collision"):
                geometry = collision.find("geometry")

                if geometry is None:
                    continue

                mesh_tag = geometry.find("mesh")

                if mesh_tag is None:
                    continue

                filename = mesh_tag.get("filename")

                if not filename:
                    continue

                mesh_path = self._resolve_mesh_uri(
                    filename
                )

                if not mesh_path.exists():
                    self.node.get_logger().warning(
                        f"Collision mesh missing: {mesh_path}"
                    )
                    continue

                loaded = trimesh.load(
                    mesh_path,
                    force="mesh",
                    process=False,
                )

                vertices = np.asarray(
                    loaded.vertices,
                    dtype=float,
                )

                faces = np.asarray(
                    loaded.faces,
                    dtype=np.int32,
                )

                if (
                    vertices.size == 0
                    or faces.size == 0
                ):
                    continue

                scale = self._parse_vector(
                    mesh_tag.get("scale"),
                    [1.0, 1.0, 1.0],
                )

                vertices = vertices * scale

                origin = collision.find("origin")

                if origin is not None:
                    xyz = self._parse_vector(
                        origin.get("xyz"),
                        [0.0, 0.0, 0.0],
                    )

                    rpy = self._parse_vector(
                        origin.get("rpy"),
                        [0.0, 0.0, 0.0],
                    )
                else:
                    xyz = np.zeros(3)
                    rpy = np.zeros(3)

                rotation = self._rpy_matrix(
                    rpy[0],
                    rpy[1],
                    rpy[2],
                )

                vertices = (
                    vertices @ rotation.T
                ) + xyz

                collision_meshes.append({
                    "link": link_name,
                    "vertices": vertices,
                    "faces": faces,
                })

        if not collision_meshes:
            raise RuntimeError(
                "NO_COLLISION_MESHES_FOUND"
            )

        self._collision_meshes = collision_meshes

        self.node.get_logger().info(
            "RobotSelfMask loaded "
            f"{len(collision_meshes)} collision meshes"
        )

    def build_mask(
        self,
        image_shape,
        fx,
        fy,
        cx,
        cy,
    ):
        if self._collision_meshes is None:
            self._load_collision_meshes()

        if timestamp is None:
            tf_time = Time()
        else:
            tf_time = Time(
                nanoseconds=int(
                    float(timestamp) * 1_000_000_000
                )
            )

        height, width = image_shape[:2]

        mask = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        for item in self._collision_meshes:
            link_name = item["link"]
            vertices = item["vertices"]

            try:
                tf = self.tf_buffer.lookup_transform(
                    self.camera_frame,
                    link_name,
                    tf_time,
                )
            except Exception as e:
                self.node.get_logger().warning(
                    f"SELF_MASK TF FAILED {link_name}: {e}"
                )
                continue

            t = tf.transform.translation
            q = tf.transform.rotation

            rotation = self._quaternion_matrix(
                q.x,
                q.y,
                q.z,
                q.w,
            )

            translation = np.array([
                t.x,
                t.y,
                t.z,
            ], dtype=float)

            camera_vertices = (
                vertices @ rotation.T
            ) + translation

            # ROS optical frame:
            # +X right, +Y down, +Z forward.
            visible = camera_vertices[:, 2] > 0.01

            camera_vertices = camera_vertices[
                visible
            ]

            if len(camera_vertices) < 3:
                continue

            z = camera_vertices[:, 2]

            u = (
                fx * camera_vertices[:, 0] / z
                + cx
            )

            v = (
                fy * camera_vertices[:, 1] / z
                + cy
            )

            points = np.column_stack([
                u,
                v,
            ])

            finite = np.all(
                np.isfinite(points),
                axis=1,
            )

            points = points[finite]

            if len(points) < 3:
                continue

            # Keep points near the image so a partially visible link
            # can still generate a conservative mask.
            margin = max(width, height)

            inside = (
                (points[:, 0] >= -margin)
                & (points[:, 0] <= width + margin)
                & (points[:, 1] >= -margin)
                & (points[:, 1] <= height + margin)
            )

            points = points[inside]

            if len(points) < 3:
                continue

            points[:, 0] = np.clip(
                points[:, 0],
                0,
                width - 1,
            )

            points[:, 1] = np.clip(
                points[:, 1],
                0,
                height - 1,
            )

            hull = cv2.convexHull(
                points.astype(np.int32)
            )

            if hull is None or len(hull) < 3:
                continue

            cv2.fillConvexPoly(
                mask,
                hull,
                255,
            )

        return mask

    def build_depth_map(
        self,
        image_shape,
        fx,
        fy,
        cx,
        cy,
        timestamp=None,
    ):
        """
        Project robot collision triangles into the camera.

        Returns a per-pixel robot depth map in metres.
        np.inf means no robot geometry projects to that pixel.

        Triangle depth uses the nearest triangle vertex,
        intentionally making the result conservative.
        """
        if self._collision_meshes is None:
            self._load_collision_meshes()

        height, width = image_shape[:2]

        depth_map = np.full(
            (height, width),
            np.inf,
            dtype=np.float32,
        )

        for item in self._collision_meshes:
            link_name = item["link"]
            vertices = item["vertices"]
            faces = item["faces"]

            try:
                tf = self.tf_buffer.lookup_transform(
                    self.camera_frame,
                    link_name,
                    Time(),
                )

            except Exception as e:
                self.node.get_logger().warning(
                    f"SELF_DEPTH TF FAILED {link_name}: {e}"
                )
                continue

            t = tf.transform.translation
            q = tf.transform.rotation

            rotation = self._quaternion_matrix(
                q.x,
                q.y,
                q.z,
                q.w,
            )

            translation = np.array(
                [t.x, t.y, t.z],
                dtype=float,
            )

            camera_vertices = (
                vertices @ rotation.T
            ) + translation

            for face in faces:
                tri = camera_vertices[face]

                # Entire triangle must be in front of camera.
                # Near-plane clipping can be added later.
                if np.any(tri[:, 2] <= 0.01):
                    continue

                z = tri[:, 2]

                u = fx * tri[:, 0] / z + cx
                v = fy * tri[:, 1] / z + cy

                points = np.column_stack(
                    [u, v]
                )

                if not np.all(
                    np.isfinite(points)
                ):
                    continue

                # Completely outside image.
                if (
                    np.max(points[:, 0]) < 0
                    or np.min(points[:, 0]) >= width
                    or np.max(points[:, 1]) < 0
                    or np.min(points[:, 1]) >= height
                ):
                    continue

                points[:, 0] = np.clip(
                    points[:, 0],
                    0,
                    width - 1,
                )

                points[:, 1] = np.clip(
                    points[:, 1],
                    0,
                    height - 1,
                )

                polygon = points.astype(
                    np.int32
                )

                triangle_mask = np.zeros(
                    (height, width),
                    dtype=np.uint8,
                )

                cv2.fillConvexPoly(
                    triangle_mask,
                    polygon,
                    255,
                )

                ys, xs = np.where(
                    triangle_mask > 0
                )

                if len(xs) == 0:
                    continue

                # Conservative Z-buffer:
                # nearest point of triangle.
                triangle_depth = float(
                    np.min(z)
                )

                current = depth_map[ys, xs]

                depth_map[ys, xs] = np.minimum(
                    current,
                    triangle_depth,
                )

        return depth_map


    @staticmethod
    def depth_aware_bbox_overlap_ratio(
        robot_depth_map,
        bbox,
        object_depth,
        depth_margin=0.015,
    ):
        """
        Fraction of bbox genuinely occluded by robot geometry
        that lies IN FRONT of the detected object.

        Robot behind target does not count as occlusion.
        """
        x1, y1, x2, y2 = [
            int(round(v))
            for v in bbox
        ]

        height, width = (
            robot_depth_map.shape[:2]
        )

        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))

        if x2 <= x1 or y2 <= y1:
            return 0.0

        if (
            object_depth is None
            or not np.isfinite(object_depth)
            or object_depth <= 0.0
        ):
            # Fail closed for unknown target depth.
            return 1.0

        region = robot_depth_map[
            y1:y2,
            x1:x2,
        ]

        if region.size == 0:
            return 0.0

        robot_present = np.isfinite(
            region
        )

        robot_in_front = (
            robot_present
            & (
                region
                < float(object_depth)
                - float(depth_margin)
            )
        )

        return float(
            np.count_nonzero(
                robot_in_front
            )
        ) / float(region.size)


    @staticmethod
    def bbox_overlap_ratio(mask, bbox):
        x1, y1, x2, y2 = [
            int(round(v))
            for v in bbox
        ]

        height, width = mask.shape[:2]

        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))

        if x2 <= x1 or y2 <= y1:
            return 0.0

        region = mask[
            y1:y2,
            x1:x2,
        ]

        if region.size == 0:
            return 0.0

        return float(
            np.count_nonzero(region)
            / region.size
        )
