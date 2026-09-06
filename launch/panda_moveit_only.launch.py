from launch import LaunchDescription, LaunchService
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():

    moveit_config = (
        MoveItConfigsBuilder(
            "moveit_resources_panda",
            package_name="moveit_resources_panda_moveit_config"
        )
        .to_moveit_configs()
    )

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {
                "use_sim_time": True,
                "allow_trajectory_execution": False,
            }
        ],
    )

    return LaunchDescription([
        move_group
    ])


if __name__ == "__main__":

    launch_service = LaunchService()

    launch_service.include_launch_description(
        generate_launch_description()
    )

    launch_service.run()
