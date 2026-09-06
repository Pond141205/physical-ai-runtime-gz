import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rclpy
import trimesh
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from tf2_ros import Buffer, TransformListener

from physical_ai_runtime.planning.viewpoint_pose import ViewpointPoseConverter


ROOT = Path(__file__).resolve().parents[2]
XACRO = ROOT / "panda_gz" / "panda_gazebo.urdf.xacro"
FINGER_MESH = Path(
    "/opt/ros/jazzy/share/moveit_resources_panda_description/"
    "meshes/collision/finger.stl"
)
HAND_FRAME = "panda_hand"
OPTICAL_FRAME = "panda_wrist_camera_optical_frame"


def rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    return Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()


def parse_xyz_rpy(origin):
    return (
        np.fromstring(origin.get("xyz"), sep=" "),
        np.fromstring(origin.get("rpy"), sep=" "),
    )


def static_hand_to_optical():
    root = ET.parse(XACRO).getroot()
    transform = np.eye(4)
    for name in (
        "panda_wrist_camera_joint",
        "panda_wrist_camera_optical_joint",
    ):
        joint = root.find(f".//joint[@name='{name}']")
        assert joint is not None
        xyz, rpy = parse_xyz_rpy(joint.find("origin"))
        step = np.eye(4)
        step[:3, :3] = rpy_matrix(rpy)
        step[:3, 3] = xyz
        transform = transform @ step
    return transform


def grasp_observation_target():
    root = ET.parse(XACRO).getroot()
    joint = root.find(".//joint[@name='panda_finger_joint1']")
    assert joint is not None
    origin, _ = parse_xyz_rpy(joint.find("origin"))
    mesh = trimesh.load(FINGER_MESH, force="mesh", process=False)
    return np.array([0.0, 0.0, origin[2] + np.mean(mesh.bounds[:, 2])])


def test_optical_axis_alignment():
    transform = static_hand_to_optical()
    target = grasp_observation_target()
    direction = target - transform[:3, 3]
    direction /= np.linalg.norm(direction)
    alignment = float(np.dot(transform[:3, 2], direction))
    print("grasp_target_hand=", target.tolist())
    print("optical_forward_hand=", transform[:3, 2].tolist())
    print("alignment=", alignment)
    assert alignment > 0.9999


def test_static_roundtrip():
    transform = static_hand_to_optical()
    converter = ViewpointPoseConverter(transform)
    camera = np.array([0.52, -0.36, 1.04])
    target = np.array([0.84, -0.36, 0.74])
    solution = converter.convert(camera, target)
    assert solution.valid
    hand = np.eye(4)
    hand[:3, :3] = Rotation.from_quat(solution.hand_quaternion_xyzw).as_matrix()
    hand[:3, 3] = solution.hand_position_world
    rebuilt = hand @ transform
    assert np.linalg.norm(rebuilt[:3, 3] - camera) < 1e-9


def test_live_tf_matches_description():
    expected = static_hand_to_optical()
    rclpy.init()
    node = rclpy.create_node("panda_wrist_camera_tf_test")
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    try:
        deadline = time.monotonic() + 5.0
        actual = None
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            try:
                actual = ViewpointPoseConverter.from_runtime_tf(
                    buffer, HAND_FRAME, OPTICAL_FRAME
                ).T_hand_optical
                break
            except RuntimeError:
                pass
        assert actual is not None, "CAMERA_EXTRINSIC_UNAVAILABLE"
        assert np.linalg.norm(actual[:3, 3] - expected[:3, 3]) < 1e-6
        assert np.linalg.norm(actual[:3, :3] - expected[:3, :3]) < 1e-6
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    test_optical_axis_alignment()
    test_static_roundtrip()
    test_live_tf_matches_description()
