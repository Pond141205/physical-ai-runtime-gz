import time
import numpy as np
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image

from physical_ai_runtime.perception.object_detection.open_vocab_detector import OpenVocabularyDetector


class GraspVerifyObserver(Node):

    def __init__(self, query="cube"):
        super().__init__("grasp_verify_observer")

        self.query = query
        self.rgb = None

        self.detector = OpenVocabularyDetector(
            box_threshold=0.20,
            text_threshold=0.20,
        )

        self.create_subscription(
            Image,
            "/runtime/grasp_verify/image",
            self.rgb_cb,
            10
        )

    def rgb_cb(self, msg):
        arr = np.frombuffer(
            msg.data,
            dtype=np.uint8
        )

        enc = msg.encoding.lower()

        if enc == "rgb8":
            self.rgb = arr.reshape(
                msg.height,
                msg.width,
                3
            ).copy()

        elif enc == "bgr8":
            img = arr.reshape(
                msg.height,
                msg.width,
                3
            )

            self.rgb = img[:, :, ::-1].copy()

    def observe_once(self, timeout=5.0):

        end = time.time() + timeout

        while rclpy.ok() and time.time() < end:

            rclpy.spin_once(
                self,
                timeout_sec=0.1
            )

            if self.rgb is None:
                continue

            detections = self.detector.detect(
                self.rgb,
                self.query
            )

            if not detections:
                continue

            best = max(
                detections,
                key=lambda d: d.confidence
            )

            return best

        return None


def main():

    rclpy.init()

    node = GraspVerifyObserver(
        query="cube"
    )

    result = node.observe_once(
        timeout=10.0
    )

    print("VERIFY DETECTION:")
    print(result)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
