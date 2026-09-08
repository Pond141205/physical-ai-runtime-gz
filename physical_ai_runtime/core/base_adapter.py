from abc import ABC, abstractmethod


class BaseRobotAdapter(ABC):

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def get_capabilities(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    def get_grasp_tool_envelope(self):
        """
        Return robot-specific grasping geometry.

        Expected keys:
            near_offset
            far_offset
            support_clearance
        """
        return None

    def get_grasp_tool_offset(self):
        """
        Return distance in meters from the tool/TCP frame
        to the nominal grasp contact region.

        Robots without a grasping tool may return None.
        """
        return None

    def get_world_up_vector_in_base(self):
        """
        Return the unit world-up vector expressed in the robot base frame.

        A grasp-capable embodiment must provide this from its frame authority
        before the common runtime can execute a semantic lift.
        """
        return None

    def get_gripper_aperture_limits(self):
        """Return semantic gripper aperture limits in meters, or None."""
        return None

    def move_gripper_aperture(
        self,
        aperture,
        *,
        tolerance,
        timeout,
    ):
        """
        Command a semantic opening between opposing fingers.

        Embodiments map this aperture to native joints and preserve native
        feedback in the returned result for contact verification.
        """
        return None
