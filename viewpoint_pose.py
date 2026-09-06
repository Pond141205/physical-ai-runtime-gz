from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from rclpy.time import Time

@dataclass(frozen=True)
class CameraPoseSolution:
    optical_position_world: np.ndarray
    optical_quaternion_xyzw: np.ndarray

    hand_position_world: np.ndarray
    hand_quaternion_xyzw: np.ndarray

    valid: bool
    reason: str


class ViewpointPoseConverter:
    """
    Converts a desired optical-camera pose into its parent tool pose.

    The hand-to-optical extrinsic is supplied by the embodiment's
    runtime TF tree. This generic geometry layer has no robot-specific
    frame names or mount constants.

    This module performs geometry only.
    It does NOT plan or execute robot motion.
    """

    def __init__(self, hand_to_optical):
        self.T_hand_optical = np.asarray(
            hand_to_optical,
            dtype=float,
        )

        if (
            self.T_hand_optical.shape != (4, 4)
            or not np.all(np.isfinite(self.T_hand_optical))
        ):
            raise ValueError("INVALID_CAMERA_EXTRINSIC")

        self.T_optical_hand = np.linalg.inv(
            self.T_hand_optical
        )

    @classmethod
    def from_runtime_tf(
        cls,
        tf_buffer,
        hand_frame,
        optical_frame,
    ):
        if not hand_frame or not optical_frame:
            raise RuntimeError("CAMERA_EXTRINSIC_UNAVAILABLE")

        try:
            transform = tf_buffer.lookup_transform(
                hand_frame,
                optical_frame,
                Time(),
            )
        except Exception as exc:
            raise RuntimeError(
                "CAMERA_EXTRINSIC_UNAVAILABLE:"
                + str(exc)
            ) from exc

        translation = transform.transform.translation
        rotation = transform.transform.rotation

        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat([
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w,
        ]).as_matrix()
        matrix[:3, 3] = [
            translation.x,
            translation.y,
            translation.z,
        ]

        return cls(matrix)

    @staticmethod
    def _normalize(v):
        v = np.asarray(
            v,
            dtype=float,
        )

        n = np.linalg.norm(v)

        if n < 1e-9:
            raise ValueError(
                "ZERO_LENGTH_VECTOR"
            )

        return v / n

    def look_at_optical_rotation(
        self,
        camera_position_world,
        target_position_world,
    ):
        """
        ROS optical frame convention:

            +Z = forward
            +X = right
            +Y = down

        Build a world->optical orientation whose +Z axis points
        toward the target.
        """

        camera = np.asarray(
            camera_position_world,
            dtype=float,
        )

        target = np.asarray(
            target_position_world,
            dtype=float,
        )

        forward_z = self._normalize(
            target - camera
        )

        world_up = np.array(
            [0.0, 0.0, 1.0],
            dtype=float,
        )

        #
        # optical +X = right
        #
        right_x = np.cross(
            forward_z,
            world_up,
        )

        if np.linalg.norm(right_x) < 1e-6:
            #
            # Top-down singularity:
            # choose deterministic world +X as optical right.
            #
            right_x = np.array(
                [1.0, 0.0, 0.0],
                dtype=float,
            )
        else:
            right_x = self._normalize(
                right_x
            )

        #
        # optical +Y = down
        #
        down_y = np.cross(
            forward_z,
            right_x,
        )

        down_y = self._normalize(
            down_y
        )

        R_world_optical = np.column_stack(
            (
                right_x,
                down_y,
                forward_z,
            )
        )

        #
        # Enforce a proper rotation matrix.
        #
        if np.linalg.det(
            R_world_optical
        ) < 0.0:
            right_x = -right_x

            R_world_optical = np.column_stack(
                (
                    right_x,
                    down_y,
                    forward_z,
                )
            )

        return R_world_optical

    def convert(
        self,
        camera_position_world,
        target_position_world,
    ) -> CameraPoseSolution:

        camera_position_world = np.asarray(
            camera_position_world,
            dtype=float,
        )

        target_position_world = np.asarray(
            target_position_world,
            dtype=float,
        )

        if (
            camera_position_world.shape != (3,)
            or target_position_world.shape != (3,)
        ):
            return CameraPoseSolution(
                optical_position_world=camera_position_world,
                optical_quaternion_xyzw=np.full(4, np.nan),
                hand_position_world=np.full(3, np.nan),
                hand_quaternion_xyzw=np.full(4, np.nan),
                valid=False,
                reason="INVALID_SHAPE",
            )

        if not (
            np.all(
                np.isfinite(
                    camera_position_world
                )
            )
            and np.all(
                np.isfinite(
                    target_position_world
                )
            )
        ):
            return CameraPoseSolution(
                optical_position_world=camera_position_world,
                optical_quaternion_xyzw=np.full(4, np.nan),
                hand_position_world=np.full(3, np.nan),
                hand_quaternion_xyzw=np.full(4, np.nan),
                valid=False,
                reason="NON_FINITE_INPUT",
            )

        try:
            R_world_optical = (
                self.look_at_optical_rotation(
                    camera_position_world,
                    target_position_world,
                )
            )

        except ValueError as exc:
            return CameraPoseSolution(
                optical_position_world=camera_position_world,
                optical_quaternion_xyzw=np.full(4, np.nan),
                hand_position_world=np.full(3, np.nan),
                hand_quaternion_xyzw=np.full(4, np.nan),
                valid=False,
                reason=str(exc),
            )

        T_world_optical = np.eye(4)

        T_world_optical[:3, :3] = (
            R_world_optical
        )

        T_world_optical[:3, 3] = (
            camera_position_world
        )

        #
        # world->hand =
        # world->optical * optical->hand
        #
        T_world_hand = (
            T_world_optical
            @ self.T_optical_hand
        )

        R_world_hand = (
            T_world_hand[:3, :3]
        )

        p_world_hand = (
            T_world_hand[:3, 3]
        )

        optical_q = Rotation.from_matrix(
            R_world_optical
        ).as_quat()

        hand_q = Rotation.from_matrix(
            R_world_hand
        ).as_quat()

        return CameraPoseSolution(
            optical_position_world=(
                camera_position_world
            ),
            optical_quaternion_xyzw=(
                optical_q
            ),
            hand_position_world=(
                p_world_hand
            ),
            hand_quaternion_xyzw=(
                hand_q
            ),
            valid=True,
            reason="POSE_CONVERSION_VALID",
        )
