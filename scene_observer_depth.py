import rclpy
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo


class SceneObserverDepth(Node):

    def __init__(self):
        super().__init__("scene_observer_depth")

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.camera_world = np.array([
            0.85,
            -0.35,
            1.40
        ])

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

        finite = np.isfinite(depth)

        mask = (
            finite &
            (depth > 0.60) &
            (depth < 0.66)
        )

        ys, xs = np.where(mask)

        if len(xs) == 0:
            print("OBJECT NOT FOUND")
            return

        u = float(xs.mean())
        v = float(ys.mean())

        z_cam = float(np.median(depth[mask]))

        x_cam = (u - self.cx) * z_cam / self.fx
        y_cam = (v - self.cy) * z_cam / self.fy

        camera_xyz = np.array([
            x_cam,
            y_cam,
            z_cam
        ])

        # Camera optical axis points downward.
        # For this top-down camera:
        #
        # camera +Z(depth) -> world -Z
        # camera +X        -> world +X
        # camera +Y        -> world -Y
        #
        world_xyz = np.array([
            self.camera_world[0] + x_cam,
            self.camera_world[1] - y_cam,
            self.camera_world[2] - z_cam
        ])

        print()
        print("OBJECT PIXELS:", len(xs))
        print("CENTROID UV:", [u, v])
        print("CAMERA XYZ:", camera_xyz)
        print("WORLD XYZ:", world_xyz)


def main():
    rclpy.init()
    node = SceneObserverDepth()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
