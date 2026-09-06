import time
from pathlib import Path

import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image


class MultiViewCapture(Node):

    TOPICS = {
        "main": "/runtime/rgbd/image",
        "side": "/runtime/grasp_verify/image",
        "wrist": "/runtime/wrist/image",
    }

    def __init__(self):
        super().__init__(
            "multiview_capture"
        )

        self.images = {}

        for name, topic in self.TOPICS.items():
            self.create_subscription(
                Image,
                topic,
                lambda msg, n=name:
                    self._image_cb(n, msg),
                10,
            )

    def _image_cb(
        self,
        name,
        msg,
    ):
        encoding = (
            msg.encoding.lower()
        )

        if encoding not in {
            "rgb8",
            "bgr8",
        }:
            return

        arr = np.frombuffer(
            msg.data,
            dtype=np.uint8,
        )

        expected = (
            msg.height
            * msg.width
            * 3
        )

        if arr.size != expected:
            return

        image = arr.reshape(
            msg.height,
            msg.width,
            3,
        )

        if encoding == "rgb8":
            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR,
            )

        self.images[name] = (
            image.copy()
        )


def capture(
    timeout=5.0,
    output_dir="/tmp/physical_ai_views",
):
    output = Path(output_dir)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    rclpy.init()

    node = MultiViewCapture()

    deadline = (
        time.time() + timeout
    )

    try:
        while (
            rclpy.ok()
            and time.time() < deadline
        ):
            rclpy.spin_once(
                node,
                timeout_sec=0.1,
            )

            if all(
                name in node.images
                for name
                in node.TOPICS
            ):
                break

        missing = [
            name
            for name
            in node.TOPICS
            if name not in node.images
        ]

        if missing:
            raise RuntimeError(
                "CAMERA_IMAGES_MISSING:"
                + ",".join(missing)
            )

        paths = {}

        for name, image in (
            node.images.items()
        ):
            path = output / (
                f"{name}.jpg"
            )

            ok = cv2.imwrite(
                str(path),
                image,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    90,
                ],
            )

            if not ok:
                raise RuntimeError(
                    f"IMAGE_WRITE_FAILED:{path}"
                )

            paths[name] = str(path)

        return paths

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    paths = capture()

    print(
        "MULTIVIEW_CAPTURED"
    )

    for name, path in (
        paths.items()
    ):
        print(
            f"{name}: {path}"
        )
