import time
import numpy as np
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo


class WorkspaceObserver(Node):

    def __init__(self):
        super().__init__("workspace_observer")

        self.depth = None
        self.fx = self.fy = None
        self.cx = self.cy = None

        self.create_subscription(
            Image,
            "/runtime/rgbd/depth_image",
            self.depth_cb,
            10
        )

        self.create_subscription(
            CameraInfo,
            "/runtime/rgbd/camera_info",
            self.info_cb,
            10
        )

    def depth_cb(self, msg):
        self.depth = np.frombuffer(
            msg.data,
            dtype=np.float32
        ).reshape(
            msg.height,
            msg.width
        ).copy()

    def info_cb(self, msg):
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

    def observe_once(self, timeout=5.0):

        end = time.time() + timeout

        while rclpy.ok() and time.time() < end:

            rclpy.spin_once(
                self,
                timeout_sec=0.1
            )

            if self.depth is None or self.fx is None:
                continue

            depth = self.depth

            valid = (
                np.isfinite(depth)
                & (depth > 0.05)
                & (depth < 3.0)
            )

            values = depth[valid]

            if len(values) < 100:
                continue

            # Histogram over observed depths.
            hist, edges = np.histogram(
                values,
                bins=300,
                range=(
                    float(values.min()),
                    float(values.max())
                )
            )

            # Ignore the globally dominant far plane (usually floor).
            order = np.argsort(hist)[::-1]

            candidate_depth = None

            for idx in order:

                center = 0.5 * (
                    edges[idx]
                    + edges[idx + 1]
                )

                count = hist[idx]

                if count < 500:
                    continue

                # Prefer a large surface closer than the far/background plane.
                if center < np.percentile(values, 80):
                    candidate_depth = float(center)
                    break

            if candidate_depth is None:
                continue

            tolerance = 0.008

            plane_mask = (
                valid
                & (
                    np.abs(
                        depth - candidate_depth
                    ) < tolerance
                )
            )

            ys, xs = np.where(plane_mask)

            if len(xs) < 500:
                continue

            # Robust image boundary of the detected support surface.
            u_min = float(np.percentile(xs, 2))
            u_max = float(np.percentile(xs, 98))
            v_min = float(np.percentile(ys, 2))
            v_max = float(np.percentile(ys, 98))

            z = float(
                np.median(
                    depth[plane_mask]
                )
            )

            def pixel_to_optical(u, v, depth_value):
                x = (
                    (u - self.cx)
                    * depth_value
                    / self.fx
                )

                y = (
                    (v - self.cy)
                    * depth_value
                    / self.fy
                )

                return np.array([
                    x,
                    y,
                    depth_value
                ])

            corners_optical = [
                pixel_to_optical(u_min, v_min, z),
                pixel_to_optical(u_max, v_min, z),
                pixel_to_optical(u_max, v_max, z),
                pixel_to_optical(u_min, v_max, z),
            ]

            #
            # Current calibrated camera extrinsic.
            # This is sensor calibration, not table geometry.
            #
            camera_world = np.array([
                0.72,
                -0.45,
                1.40
            ])

            rotation_camera_to_world = np.array([
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ])

            corners_world = []

            for optical in corners_optical:

                xo, yo, zo = optical

                camera_local = np.array([
                    zo,
                    -xo,
                    -yo
                ])

                world = (
                    camera_world
                    + rotation_camera_to_world
                    @ camera_local
                )

                corners_world.append(world)

            corners_world = np.array(
                corners_world
            )

            x_min = float(
                np.min(corners_world[:, 0])
            )

            x_max = float(
                np.max(corners_world[:, 0])
            )

            y_min = float(
                np.min(corners_world[:, 1])
            )

            y_max = float(
                np.max(corners_world[:, 1])
            )

            table_z = float(
                np.median(
                    corners_world[:, 2]
                )
            )

            return {
                "plane_depth": z,
                "table_z": table_z,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "width": x_max - x_min,
                "length": y_max - y_min,
                "pixels": len(xs),
            }

        return None


def main():

    rclpy.init()

    node = WorkspaceObserver()

    result = node.observe_once()

    print("WORKSPACE:")
    print(result)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
