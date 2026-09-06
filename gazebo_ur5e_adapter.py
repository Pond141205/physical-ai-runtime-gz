import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


class GazeboUR5eAdapter(Node):

    def __init__(self):
        super().__init__("gazebo_ur5e_adapter")

        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        self.current_joint_position = None

        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10
        )

        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/scaled_joint_trajectory_controller/follow_joint_trajectory"
        )

    def joint_state_callback(self, msg):
        joint_map = dict(zip(msg.name, msg.position))

        try:
            self.current_joint_position = np.array([
                joint_map[name]
                for name in self.joint_names
            ])
        except KeyError:
            return

    def wait_for_joint_state(self, timeout=5.0):
        start = time.time()

        while rclpy.ok() and self.current_joint_position is None:
            rclpy.spin_once(
                self,
                timeout_sec=0.1
            )

            if time.time() - start > timeout:
                raise RuntimeError(
                    "JOINT_STATE_TIMEOUT"
                )

    def move_joints(self, target_q, duration=3.0):
        target_q = np.array(
            target_q,
            dtype=float
        )

        self.wait_for_joint_state()

        if not self.trajectory_client.wait_for_server(
            timeout_sec=5.0
        ):
            raise RuntimeError(
                "TRAJECTORY_ACTION_NOT_AVAILABLE"
            )

        goal = FollowJointTrajectory.Goal()

        goal.trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = target_q.tolist()

        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int(
            (duration - int(duration)) * 1e9
        )

        goal.trajectory.points = [point]

        future = self.trajectory_client.send_goal_async(
            goal
        )

        rclpy.spin_until_future_complete(
            self,
            future
        )

        goal_handle = future.result()

        if not goal_handle.accepted:
            raise RuntimeError(
                "TRAJECTORY_GOAL_REJECTED"
            )

        result_future = goal_handle.get_result_async()

        rclpy.spin_until_future_complete(
            self,
            result_future
        )

        return result_future.result().result
