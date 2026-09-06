from launch import LaunchDescription, LaunchService
from launch.actions import ExecuteProcess, TimerAction, SetEnvironmentVariable, RegisterEventHandler
from launch.substitutions import Command
from launch.event_handlers import OnProcessExit
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node


UR_XACRO = "/home/pond/physical-ai-runtime-gz/ur5e_gz/ur5e_dual.urdf.xacro"

UR_CONTROLLERS = (
    "/home/pond/physical-ai-runtime-gz/"
    "ur5e_gz/ur_controllers_dual.yaml"
)

PANDA_XACRO = (
    "/home/pond/physical-ai-runtime-gz/"
    "panda_gz/panda_gazebo.urdf.xacro"
)

PANDA_JOINT_STATE_MERGER = (
    "/home/pond/physical-ai-runtime-gz/"
    "physical_ai_runtime/adapters/panda_joint_state_merger.py"
)


def generate_launch_description():

    # ==========================================================
    # Robot descriptions
    # ==========================================================

    ur_description = Command([
        "xacro ",
        UR_XACRO,
        " ",
        "name:=ur5e ",
        "ur_type:=ur5e ",
        "tf_prefix:= ",
        "simulation_controllers:=",
        UR_CONTROLLERS,
        " ",
        "ros_namespace:=/ur5e"
    ])

    panda_description = Command([
        "xacro ",
        PANDA_XACRO
    ])

    # ==========================================================
    # Gazebo
    # ==========================================================

    gazebo = ExecuteProcess(
        cmd=[
            "gz",
            "sim",
            "-r",
            "worlds/runtime_world.sdf"
        ],
        output="screen"
    )

    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"
        ],
        output="screen"
    )

    # ==========================================================
    # Robot state publishers
    # ==========================================================

    ur_rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace="ur5e",
        name="robot_state_publisher",
        parameters=[
            {
                "use_sim_time": True,
                "robot_description": ParameterValue(
                    ur_description,
                    value_type=str
                )
            }
        ],
        output="screen"
    )

    panda_rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace="panda",
        name="robot_state_publisher",
        parameters=[
            {
                "use_sim_time": True,
                "robot_description": ParameterValue(
                    panda_description,
                    value_type=str
                )
            }
        ],
        remappings=[
            (
                "joint_states",
                "/panda/robot_joint_states",
            ),
            (
                "/robot_description",
                "/panda/robot_description"
            )
        ],
        output="screen"
    )
    panda_joint_state_merger = ExecuteProcess(
        cmd=[
            "python3",
            PANDA_JOINT_STATE_MERGER,
        ],
        output="screen"
    )


    # ==========================================================
    # Spawn robots
    # ==========================================================

    spawn_ur = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-string",
            ur_description,
            "-name",
            "ur5e"
        ],
        output="screen"
    )

    spawn_panda = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-string",
            panda_description,
            "-name",
            "panda"
        ],
        output="screen"
    )


    # ==========================================================
    # Perception scene
    # ==========================================================

    # Camera poses are defined once here and reused by both
    # Gazebo spawning and TF publication.
    main_cam_xyz = (0.72, -0.45, 1.40)
    main_cam_rpy = (0.0, 1.5708, 0.0)

    verify_cam_xyz = (0.72, -1.05, 0.95)
    verify_cam_rpy = (0.0, 0.550375, 1.5708)


    spawn_camera = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-file",
            "/home/pond/physical-ai-runtime-gz/sensors/rgbd_camera.sdf",
            "-name",
            "runtime_rgbd_camera",
            "-x",
            str(main_cam_xyz[0]),
            "-y",
            str(main_cam_xyz[1]),
            "-z",
            str(main_cam_xyz[2]),
            "-R",
            str(main_cam_rpy[0]),
            "-P",
            str(main_cam_rpy[1]),
            "-Y",
            str(main_cam_rpy[2])
        ],
        output="screen"
    )

    spawn_verify_camera = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-file",
            "/home/pond/physical-ai-runtime-gz/sensors/grasp_verify_camera.sdf",
            "-name",
            "grasp_verify_camera",
            "-x",
            str(verify_cam_xyz[0]),
            "-y",
            str(verify_cam_xyz[1]),
            "-z",
            str(verify_cam_xyz[2]),
            "-R",
            str(verify_cam_rpy[0]),
            "-P",
            str(verify_cam_rpy[1]),
            "-Y",
            str(verify_cam_rpy[2])
        ],
        output="screen"
    )

    main_camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x", str(main_cam_xyz[0]),
            "--y", str(main_cam_xyz[1]),
            "--z", str(main_cam_xyz[2]),
            "--roll", str(main_cam_rpy[0]),
            "--pitch", str(main_cam_rpy[1]),
            "--yaw", str(main_cam_rpy[2]),
            "--frame-id", "world",
            "--child-frame-id", "runtime_rgbd_camera_link",
        ],
        output="screen",
    )

    main_camera_optical_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x", "0",
            "--y", "0",
            "--z", "0",
            "--roll", "-1.57079632679",
            "--pitch", "0",
            "--yaw", "-1.57079632679",
            "--frame-id", "runtime_rgbd_camera_link",
            "--child-frame-id", "runtime_rgbd_camera_optical_frame",
        ],
        output="screen",
    )

    verify_camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x", str(verify_cam_xyz[0]),
            "--y", str(verify_cam_xyz[1]),
            "--z", str(verify_cam_xyz[2]),
            "--roll", str(verify_cam_rpy[0]),
            "--pitch", str(verify_cam_rpy[1]),
            "--yaw", str(verify_cam_rpy[2]),
            "--frame-id", "world",
            "--child-frame-id", "grasp_verify_camera_link",
        ],
        output="screen",
    )

    verify_camera_optical_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x", "0",
            "--y", "0",
            "--z", "0",
            "--roll", "-1.57079632679",
            "--pitch", "0",
            "--yaw", "-1.57079632679",
            "--frame-id", "grasp_verify_camera_link",
            "--child-frame-id", "grasp_verify_camera_optical_frame",
        ],
        output="screen",
    )


    spawn_table = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-file",
            "/home/pond/physical-ai-runtime-gz/test_objects/grasp_table.sdf",
            "-name",
            "grasp_table",
            "-x",
            "0.72",
            "-y",
            "-0.45",
            "-z",
            "0.70"
        ],
        output="screen"
    )

    spawn_cube = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-file",
            "/home/pond/physical-ai-runtime-gz/test_objects/grasp_cube.sdf",
            "-name",
            "grasp_cube_table",
            "-x",
            "0.85",
            "-y",
            "-0.35",
            "-z",
            "0.745"
        ],
        output="screen"
    )

    rgbd_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/runtime/rgbd/image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/runtime/rgbd/depth_image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/runtime/rgbd/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
            "/runtime/rgbd/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked",
        ],
        output="screen"
    )

    grasp_verify_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/runtime/grasp_verify/image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/runtime/grasp_verify/depth_image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/runtime/grasp_verify/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
            "/runtime/grasp_verify/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked",
        ],
        output="screen"
    )

    # ==========================================================
    # UR controllers
    # ==========================================================

    ur_jsb = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "-c",
            "/ur5e/controller_manager"
        ],
        output="screen"
    )

    ur_arm = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "scaled_joint_trajectory_controller",
            "-c",
            "/ur5e/controller_manager"
        ],
        output="screen"
    )

    # ==========================================================
    # Panda controllers
    # ==========================================================

    panda_jsb = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "-c",
            "/panda/controller_manager"
        ],
        output="screen"
    )

    panda_arm = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "panda_arm_controller",
            "-c",
            "/panda/controller_manager"
        ],
        output="screen"
    )

    # ==========================================================
    # Sequential controller startup
    # ==========================================================

    start_ur_arm_after_ur_jsb = RegisterEventHandler(
        OnProcessExit(
            target_action=ur_jsb,
            on_exit=[
                ur_arm
            ]
        )
    )

    start_panda_jsb_after_ur_arm = RegisterEventHandler(
        OnProcessExit(
            target_action=ur_arm,
            on_exit=[
                panda_jsb
            ]
        )
    )

    start_panda_arm_after_panda_jsb = RegisterEventHandler(
        OnProcessExit(
            target_action=panda_jsb,
            on_exit=[
                panda_arm
            ]
        )
    )

    top_camera_view = ExecuteProcess(
        cmd=[
            "ros2",
            "run",
            "rqt_image_view",
            "rqt_image_view",
            "/runtime/rgbd/image"
        ],
        output="screen"
    )

    verify_camera_view = ExecuteProcess(
        cmd=[
            "ros2",
            "run",
            "rqt_image_view",
            "rqt_image_view",
            "/runtime/grasp_verify/image"
        ],
        output="screen"
    )


    wrist_camera_view = ExecuteProcess(
        cmd=[
            "ros2",
            "run",
            "rqt_image_view",
            "rqt_image_view",
            "/runtime/wrist/image"
        ],
        output="screen"
    )

    return LaunchDescription([

        SetEnvironmentVariable(
            "GZ_SIM_SYSTEM_PLUGIN_PATH",
            "/opt/ros/jazzy/lib"
        ),

        gazebo,
        clock_bridge,

        TimerAction(
            period=5.0,
            actions=[
                top_camera_view,
                verify_camera_view,
                wrist_camera_view
            ]
        ),

        TimerAction(
            period=2.0,
            actions=[
                ur_rsp,
                panda_joint_state_merger,
                panda_rsp,
                spawn_ur,
                spawn_panda
            ]
        ),

        start_ur_arm_after_ur_jsb,
        start_panda_jsb_after_ur_arm,
        start_panda_arm_after_panda_jsb,

        TimerAction(
            period=3.0,
            actions=[
                spawn_camera,
                spawn_verify_camera,
        verify_camera_optical_tf,
        verify_camera_tf,
        main_camera_optical_tf,
        main_camera_tf,
                spawn_table,
                spawn_cube,
                rgbd_bridge,
                grasp_verify_bridge
            ]
        ),

        TimerAction(
            period=6.0,
            actions=[
                ur_jsb
            ]
        ),
    ])



if __name__ == "__main__":

    service = LaunchService()

    service.include_launch_description(
        generate_launch_description()
    )

    service.run()
