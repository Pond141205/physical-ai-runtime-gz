import time

import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from tf2_ros import Buffer, TransformListener

from robot_self_mask import RobotSelfMask


class MaskTest(Node):

    def __init__(self):
        super().__init__("robot_self_mask_test")

        self.rgb = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        self.create_subscription(
            Image,
            "/runtime/rgbd/image",
            self.rgb_cb,
            10
        )

        self.create_subscription(
            CameraInfo,
            "/runtime/rgbd/camera_info",
            self.info_cb,
            10
        )

        self.self_mask = RobotSelfMask(
            node=self,
            tf_buffer=self.tf_buffer,
            camera_frame="runtime_rgbd_camera_optical_frame",
            robot_description_node="/panda/robot_state_publisher",
        )

    def rgb_cb(self, msg):
        arr = np.frombuffer(
            msg.data,
            dtype=np.uint8
        )

        if msg.encoding.lower() == "rgb8":
            self.rgb = arr.reshape(
                msg.height,
                msg.width,
                3
            ).copy()

        elif msg.encoding.lower() == "bgr8":
            img = arr.reshape(
                msg.height,
                msg.width,
                3
            )
            self.rgb = img[:, :, ::-1].copy()

    def info_cb(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]


def main():
    rclpy.init()

    node = MaskTest()

    end_time = time.time() + 10.0

    while rclpy.ok() and time.time() < end_time:
        rclpy.spin_once(
            node,
            timeout_sec=0.1
        )

        if (
            node.rgb is None
            or node.fx is None
        ):
            continue

        mask = node.self_mask.build_mask(
            image_shape=node.rgb.shape,
            fx=node.fx,
            fy=node.fy,
            cx=node.cx,
            cy=node.cy,
        )

        rgb_bgr = cv2.cvtColor(
            node.rgb,
            cv2.COLOR_RGB2BGR
        )

        overlay = rgb_bgr.copy()

        overlay[mask > 0] = (
            0,
            0,
            255
        )

        result = cv2.addWeighted(
            rgb_bgr,
            0.6,
            overlay,
            0.4,
            0.0
        )

        cv2.imwrite(
            "/tmp/robot_self_mask.png",
            result
        )

        cv2.imwrite(
            "/tmp/robot_self_mask_binary.png",
            mask
        )

        print(
            "MASK PIXELS:",
            int(np.count_nonzero(mask))
        )

        print(
            "SAVED:",
            "/tmp/robot_self_mask.png"
        )

        break

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
