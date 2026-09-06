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
