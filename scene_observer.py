import rclpy
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
from scipy.ndimage import label

from scene_state import SceneObject, SceneState


class SceneObserver(Node):

    def __init__(self):
        super().__init__("scene_observer")

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.latest_scene = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        self.create_subscription(
            CameraInfo,
            "/runtime/rgbd/camera_info",
            self.info_cb,
            10
        )

        self.create_subscription(
            Image,
            "/runtime/rgbd/depth_image",
            self.depth_cb,
            10
        )

    def info_cb(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def depth_cb(self, msg):

        if self.fx is None:
            return

        depth = np.frombuffer(
            msg.data,
            dtype=np.float32
        ).reshape(msg.height, msg.width)

        mask = (
            np.isfinite(depth)
            & (depth > 0.60)
            & (depth < 0.66)
        )

        labels, count = label(mask)

        candidates = []

        for i in range(1, count + 1):

            ys, xs = np.where(labels == i)

            pixels = len(xs)

            if pixels < 100:
                continue

            u = float(xs.mean())
            v = float(ys.mean())

            z_cam = float(
                np.median(depth[ys, xs])
            )

            x_cam = (
                (u - self.cx)
                * z_cam
                / self.fx
            )

            y_cam = (
                (v - self.cy)
                * z_cam
                / self.fy
            )

            camera_point = PointStamped()
            camera_point.header.frame_id = (
                "runtime_rgbd_camera/camera_link/rgbd"
            )

            camera_point.point.x = float(x_cam)
            camera_point.point.y = float(y_cam)
            camera_point.point.z = float(z_cam)

            try:
                camera_to_world = self.tf_buffer.lookup_transform(
                    "world",
                    camera_point.header.frame_id,
                    rclpy.time.Time()
                )

                world_point = do_transform_point(
                    camera_point,
                    camera_to_world
                )

                world_xyz = np.array([
                    world_point.point.x,
                    world_point.point.y,
                    world_point.point.z
                ])

            except Exception:
                continue

            candidates.append({
                "pixels": pixels,
                "pixel": [u, v],
                "depth": z_cam,
                "position_world": world_xyz
            })

        if not candidates:
            return

        candidates.sort(
            key=lambda c:
            (c["pixel"][0] - self.cx) ** 2
            + (c["pixel"][1] - self.cy) ** 2
        )

        obj = candidates[0]

        world_point = PointStamped()
        world_point.header.frame_id = "world"

        world_point.point.x = float(
            obj["position_world"][0]
        )
        world_point.point.y = float(
            obj["position_world"][1]
        )
        world_point.point.z = float(
            obj["position_world"][2]
        )

        try:
            tf = self.tf_buffer.lookup_transform(
                "panda_link0",
                "world",
                rclpy.time.Time()
            )

            robot_point = do_transform_point(
                world_point,
                tf
            )

        except Exception:
            return

        robot_xyz = np.array([
            robot_point.point.x,
            robot_point.point.y,
            robot_point.point.z
        ])

        scene_object = SceneObject(
            object_id="cube",
            position_world=np.array(
                obj["position_world"],
                dtype=float
            ),
            position_robot=robot_xyz,
            depth=float(obj["depth"])
        )

        self.latest_scene = SceneState(
            objects={
                "cube": scene_object
            }
        )

    def observe_once(self, timeout=5.0):

        self.latest_scene = None

        start = self.get_clock().now()

        while rclpy.ok():

            rclpy.spin_once(
                self,
                timeout_sec=0.1
            )

            if self.latest_scene is not None:
                return self.latest_scene

            elapsed = (
                self.get_clock().now()
                - start
            ).nanoseconds / 1e9

            if elapsed > timeout:
                raise RuntimeError(
                    "SCENE_OBSERVATION_TIMEOUT"
                )


def main():

    rclpy.init()

    node = SceneObserver()

    try:
        scene = node.observe_once()

        print("SCENE:")
        print(scene)

        cube = scene.objects["cube"]

        print("CUBE WORLD:")
        print(cube.position_world)

        print("CUBE ROBOT:")
        print(cube.position_robot)

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
