from launch import LaunchDescription, LaunchService
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from pathlib import Path

def generate_launch_description():

    # =========================
    # Panda MoveIt
    # =========================

    panda_config = (
        MoveItConfigsBuilder(
            "moveit_resources_panda",
            package_name="moveit_resources_panda_moveit_config"
        )
        .robot_description(
            file_path=Path(
                "/home/pond/physical-ai-runtime-gz/"
                "panda_gz/panda_gazebo.urdf.xacro"
            ),
            mappings={
                "with_wrist_camera": "true"
            }
        )
        .to_moveit_configs()
    )

    panda_move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        namespace="panda",
        name="move_group",
        output="screen",
        parameters=[
            panda_config.to_dict(),
            {
                "use_sim_time": True,
                "allow_trajectory_execution": False,
            }
        ],
    )

    # =========================
    # UR5e MoveIt
    # =========================

    ur_config = (
        MoveItConfigsBuilder(
            robot_name="ur",
            package_name="ur_moveit_config"
        )
        .robot_description_semantic(
            Path("srdf") / "ur.srdf.xacro",
            {
                "name": "ur5e"
            }
        )
        .robot_description_kinematics(
            file_path="config/kinematics.yaml"
        )
        .joint_limits(
            file_path="config/joint_limits.yaml"
        )
        .trajectory_execution(
            file_path="config/moveit_controllers.yaml"
        )
        .to_moveit_configs()
    )

    ur_move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        namespace="ur5e",
        name="move_group",
        output="screen",
        parameters=[
            ur_config.to_dict(),
            {
                "use_sim_time": True,
                "allow_trajectory_execution": False,
            }
        ],
        remappings=[
            ("robot_description", "/ur5e/robot_description")
        ]
    )

    return LaunchDescription([
        panda_move_group,
        ur_move_group
    ])


if __name__ == "__main__":

    service = LaunchService()

    service.include_launch_description(
        generate_launch_description()
    )

    service.run()
