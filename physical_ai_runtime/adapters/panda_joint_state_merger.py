import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class PandaJointStateMerger(Node):
    def __init__(self):
        super().__init__(
            "panda_joint_state_merger"
        )

        self.arm_msg = None
        self.gripper_msg = None

        self.last_arm_stamp_ns = None
        self.last_gripper_stamp_ns = None
        self.last_publish_stamp_ns = None

        # Maximum acceptable sensor-time separation
        # between arm and gripper state.
        self.max_pair_skew_ns = int(
            0.10 * 1_000_000_000
        )

        self.pub = self.create_publisher(
            JointState,
            "/panda/robot_joint_states",
            10,
        )

        self.create_subscription(
            JointState,
            "/panda/joint_states",
            self.arm_cb,
            10,
        )

        self.create_subscription(
            JointState,
            "/panda_gripper_joint_states",
            self.gripper_cb,
            10,
        )

    @staticmethod
    def stamp_ns(msg):
        stamp = msg.header.stamp

        return (
            int(stamp.sec) * 1_000_000_000
            + int(stamp.nanosec)
        )

    def reset_cache(
        self,
        reason,
    ):
        self.get_logger().warning(
            f"MERGER CACHE RESET: {reason}"
        )

        self.arm_msg = None
        self.gripper_msg = None

        self.last_arm_stamp_ns = None
        self.last_gripper_stamp_ns = None
        self.last_publish_stamp_ns = None

    def detect_time_reset(
        self,
        stamp_ns,
        previous_stamp_ns,
        stream_name,
    ):
        if previous_stamp_ns is None:
            return False

        if stamp_ns < previous_stamp_ns:
            self.reset_cache(
                f"{stream_name} time moved backwards "
                f"{previous_stamp_ns} -> {stamp_ns}"
            )
            return True

        return False

    def arm_cb(self, msg):
        stamp = self.stamp_ns(msg)

        self.detect_time_reset(
            stamp,
            self.last_arm_stamp_ns,
            "arm",
        )

        self.arm_msg = msg
        self.last_arm_stamp_ns = stamp

        self.publish_merged()

    def gripper_cb(self, msg):
        stamp = self.stamp_ns(msg)

        self.detect_time_reset(
            stamp,
            self.last_gripper_stamp_ns,
            "gripper",
        )

        self.gripper_msg = msg
        self.last_gripper_stamp_ns = stamp

        self.publish_merged()

    def publish_merged(self):
        if (
            self.arm_msg is None
            or self.gripper_msg is None
        ):
            return

        arm_ns = self.stamp_ns(
            self.arm_msg
        )

        gripper_ns = self.stamp_ns(
            self.gripper_msg
        )

        skew_ns = abs(
            arm_ns - gripper_ns
        )

        # Do not merge states that are too far apart
        # in simulation / sensor time.
        if skew_ns > self.max_pair_skew_ns:
            return

        publish_ns = max(
            arm_ns,
            gripper_ns,
        )

        # Prevent publishing time that moves backwards.
        if (
            self.last_publish_stamp_ns
            is not None
            and publish_ns
            < self.last_publish_stamp_ns
        ):
            self.reset_cache(
                "merged output time moved backwards"
            )
            return

        out = JointState()

        if arm_ns >= gripper_ns:
            out.header.stamp = (
                self.arm_msg.header.stamp
            )
        else:
            out.header.stamp = (
                self.gripper_msg.header.stamp
            )

        out.name = (
            list(self.arm_msg.name)
            + list(self.gripper_msg.name)
        )

        out.position = (
            list(self.arm_msg.position)
            + list(self.gripper_msg.position)
        )

        out.velocity = (
            list(self.arm_msg.velocity)
            + list(self.gripper_msg.velocity)
        )

        out.effort = (
            list(self.arm_msg.effort)
            + list(self.gripper_msg.effort)
        )

        self.pub.publish(out)

        self.last_publish_stamp_ns = (
            publish_ns
        )


def main():
    rclpy.init()

    node = PandaJointStateMerger()

    try:
        rclpy.spin(node)

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
