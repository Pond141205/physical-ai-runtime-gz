from launch import LaunchDescription, LaunchService
from launch.actions import (
    IncludeLaunchDescription,
    ExecuteProcess,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource


ROBOT_LAUNCH = (
    "/home/pond/physical-ai-runtime-gz/"
    "launch/dual_robot.launch.py"
)

MOVEIT_LAUNCH = (
    "/home/pond/physical-ai-runtime-gz/"
    "launch/dual_moveit.launch.py"
)

PROJECT_ROOT = "/home/pond/physical-ai-runtime-gz"
VENV_PYTHON = PROJECT_ROOT + "/.venv/bin/python"


def generate_launch_description():

    robots = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            ROBOT_LAUNCH
        )
    )

    wait_controllers = ExecuteProcess(
        cmd=[
            "bash",
            "-lc",
            r'''
echo "Waiting for dual controllers..."

while true; do

    UR="$(ros2 control list_controllers \
        -c /ur5e/controller_manager 2>/dev/null || true)"

    PANDA="$(ros2 control list_controllers \
        -c /panda/controller_manager 2>/dev/null || true)"

    if echo "$UR" | grep -q \
        "joint_state_broadcaster.*active" && \
       echo "$UR" | grep -q \
        "scaled_joint_trajectory_controller.*active" && \
       echo "$PANDA" | grep -q \
        "joint_state_broadcaster.*active" && \
       echo "$PANDA" | grep -q \
        "panda_arm_controller.*active"; then

        echo "UR5e controllers READY"
        echo "Panda controllers READY"
        exit 0
    fi

    sleep 1
done
'''
        ],
        output="screen"
    )

    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            MOVEIT_LAUNCH
        )
    )

    wait_moveit = ExecuteProcess(
        cmd=[
            "bash",
            "-lc",
            r'''
echo "Waiting for MoveIt IK services..."

while true; do

    SERVICES="$(ros2 service list 2>/dev/null || true)"

    if echo "$SERVICES" | grep -qx \
        "/ur5e/compute_ik" && \
       echo "$SERVICES" | grep -qx \
        "/panda/compute_ik"; then

        echo ""
        echo "========================================"
        echo " PHYSICAL AI RUNTIME READY"
        echo "========================================"
        echo " UR5e  controller : READY"
        echo " Panda controller : READY"
        echo " UR5e  MoveIt IK  : READY"
        echo " Panda MoveIt IK  : READY"
        echo "========================================"
        exit 0
    fi

    sleep 1
done
'''
        ],
        output="screen"
    )

    gripper_command_bridge = ExecuteProcess(
        cmd=[
            "ros2",
            "run",
            "ros_gz_bridge",
            "parameter_bridge",
            "/panda_gripper_native_trajectory@trajectory_msgs/msg/JointTrajectory@gz.msgs.JointTrajectory",
        ],
        output="screen"
    )

    gripper_state_bridge = ExecuteProcess(
        cmd=[
            "ros2",
            "run",
            "ros_gz_bridge",
            "parameter_bridge",
            "/panda_gripper_joint_states@sensor_msgs/msg/JointState@gz.msgs.Model",
        ],
        output="screen"
    )


    # =========================
    # Panda wrist RGB-D camera
    # =========================

    wrist_camera_bridge = ExecuteProcess(
        cmd=[
            "ros2",
            "run",
            "ros_gz_bridge",
            "parameter_bridge",
            "/runtime/wrist/image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/runtime/wrist/depth_image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/runtime/wrist/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
            "/runtime/wrist/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked",
        ],
        output="screen"
    )

    start_moveit = RegisterEventHandler(
        OnProcessExit(
            target_action=wait_controllers,
            on_exit=[
                moveit,
                TimerAction(
                    period=1.0,
                    actions=[wait_moveit]
                )
            ]
        )
    )


    workspace_scene_sync = ExecuteProcess(
        cmd=[
            "bash",
            "-lc",
            rf"""
cd {PROJECT_ROOT}

echo ""
echo "Waiting for PlanningScene workspace synchronization..."

for ATTEMPT in $(seq 1 15); do

    echo ""
    echo "PlanningScene sync attempt $ATTEMPT/15"

    OUTPUT="$({VENV_PYTHON} workspace_scene_sync.py 2>&1)"
    STATUS=$?

    echo "$OUTPUT"

    if echo "$OUTPUT" | grep -q         "PLANNING SCENE APPLIED: True"; then

        echo ""
        echo "========================================"
        echo " PLANNING SCENE SAFETY CONTEXT READY"
        echo "========================================"
        echo " detected_support_surface : READY"
        echo " Wrist RGB-D              : ENABLED"
        echo "========================================"
        exit 0
    fi

    sleep 2
done

echo ""
echo "========================================"
echo " PLANNING SCENE SYNC FAILED"
echo " Autonomous motion remains FAIL-CLOSED."
echo "========================================"

exit 1
"""
        ],
        output="screen"
    )

    start_workspace_sync = RegisterEventHandler(
        OnProcessExit(
            target_action=wait_moveit,
            on_exit=[
                TimerAction(
                    period=1.0,
                    actions=[workspace_scene_sync]
                )
            ]
        )
    )

    return LaunchDescription([
        robots,

        gripper_command_bridge,
        gripper_state_bridge,
        wrist_camera_bridge,

        start_moveit,
        start_workspace_sync,

        TimerAction(
            period=1.0,
            actions=[wait_controllers]
        ),
    ])


if __name__ == "__main__":

    service = LaunchService()

    service.include_launch_description(
        generate_launch_description()
    )

    service.run()
