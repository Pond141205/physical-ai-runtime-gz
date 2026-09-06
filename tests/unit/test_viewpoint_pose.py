import numpy as np
from scipy.spatial.transform import Rotation

from physical_ai_runtime.planning.viewpoint_pose import (
    ViewpointPoseConverter,
)


C = "\033[1;36m"
G = "\033[1;32m"
Y = "\033[1;33m"
W = "\033[0m"


camera = np.array([
    0.5287153011,
    -0.3597458251,
    1.0437657319,
])

target = np.array([
    0.8487153011,
    -0.3597458251,
    0.7437657319,
])


hand_to_optical = np.eye(4)
hand_to_optical[:3, :3] = Rotation.from_euler(
    "ZYX",
    [-1.57079632679, 0.0, -1.57079632679],
).as_matrix()
hand_to_optical[:3, :3] = Rotation.from_euler(
    "y",
    -2.56247811,
).as_matrix() @ hand_to_optical[:3, :3]
hand_to_optical[:3, 3] = [
    0.100,
    0.0,
    0.020,
]
converter = ViewpointPoseConverter(hand_to_optical)

solution = converter.convert(
    camera,
    target,
)


print(
    f"\n{C}"
    "════════ VIEWPOINT POSE CONVERSION ════════"
    f"{W}"
)

print(
    f"{Y}valid:{W}",
    solution.valid,
)

print(
    f"{Y}reason:{W}",
    solution.reason,
)

print(
    f"{Y}optical position:{W}",
    solution
    .optical_position_world
    .round(6)
    .tolist(),
)

print(
    f"{Y}optical quaternion xyzw:{W}",
    solution
    .optical_quaternion_xyzw
    .round(6)
    .tolist(),
)

print(
    f"{Y}hand position:{W}",
    solution
    .hand_position_world
    .round(6)
    .tolist(),
)

print(
    f"{Y}hand quaternion xyzw:{W}",
    solution
    .hand_quaternion_xyzw
    .round(6)
    .tolist(),
)


assert solution.valid

#
# Forward direction of optical +Z must point to target.
#
from scipy.spatial.transform import Rotation

R = Rotation.from_quat(
    solution.optical_quaternion_xyzw
).as_matrix()

forward = R[:, 2]

expected = (
    target - camera
)

expected = (
    expected
    / np.linalg.norm(expected)
)

alignment = float(
    np.dot(
        forward,
        expected,
    )
)

print(
    f"{Y}look-at alignment:{W}",
    alignment,
)

assert alignment > 0.999999


#
# Reconstruct optical pose from generated hand pose
# and known hand->optical transform.
#
T_world_hand = np.eye(4)

T_world_hand[:3, :3] = (
    Rotation.from_quat(
        solution.hand_quaternion_xyzw
    ).as_matrix()
)

T_world_hand[:3, 3] = (
    solution.hand_position_world
)

T_world_optical_check = (
    T_world_hand
    @ converter.T_hand_optical
)

position_error = float(
    np.linalg.norm(
        T_world_optical_check[:3, 3]
        - camera
    )
)

rotation_error = float(
    np.linalg.norm(
        T_world_optical_check[:3, :3]
        - R
    )
)

print(
    f"{Y}roundtrip position error:{W}",
    position_error,
)

print(
    f"{Y}roundtrip rotation error:{W}",
    rotation_error,
)


assert position_error < 1e-9
assert rotation_error < 1e-9


print(
    f"\n{G}"
    "✓ OPTICAL → HAND CONVERSION PASSED"
    f"{W}"
)

print(
    "\nGEOMETRY ONLY — "
    "NO MOVEIT PLAN — "
    "NO ROBOT MOTION"
)
