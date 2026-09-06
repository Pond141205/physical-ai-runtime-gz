from dataclasses import dataclass


@dataclass
class RobotCapabilities:
    move_tcp: bool
    grasp: bool
    force_control: bool
    locomotion: bool
    move_joints: bool = False
