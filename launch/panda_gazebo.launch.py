from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():

    gz = ExecuteProcess(
        cmd=[
            "gz",
            "sim",
            "-r",
            "empty.sdf"
        ],
        output="screen"
    )

    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name",
            "panda",
            "-file",
            "/home/pond/physical-ai-runtime-gz/panda_gz/panda_gazebo.urdf",
            "-x",
            "0",
            "-y",
            "0",
            "-z",
            "0"
        ],
        output="screen"
    )

    return LaunchDescription([
        gz,
        spawn
    ])
