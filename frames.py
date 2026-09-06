import numpy as np


class FrameTransform:

    def __init__(self):
        self.workspace_origins = {}

    def register_robot(
        self,
        robot_name,
        origin
    ):
        self.workspace_origins[
            robot_name
        ] = np.array(
            origin,
            dtype=float
        )

    def to_robot_base(
        self,
        robot_name,
        position,
        frame
    ):
        position = np.array(
            position,
            dtype=float
        )

        if frame == "robot_base":
            return position

        if frame == "workspace":

            if robot_name not in self.workspace_origins:
                raise RuntimeError(
                    "WORKSPACE_FRAME_NOT_REGISTERED"
                )

            return (
                self.workspace_origins[
                    robot_name
                ]
                + position
            )

        raise RuntimeError(
            f"UNKNOWN_FRAME:{frame}"
        )