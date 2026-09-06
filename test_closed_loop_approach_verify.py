import numpy as np
import rclpy

from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.msg import (
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
)
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose

from gazebo_panda_adapter import GazeboPandaAdapter
from runtime import RobotRuntime
from skills import GraspObject
from rgbd_scene_observer import RGBDSceneObserver
from workspace_observer import WorkspaceObserver


def plan_tcp(robot, target, quat):
    request = GetMotionPlan.Request()
    req = request.motion_plan_request

    req.group_name = robot.move_group
    req.num_planning_attempts = 5
    req.allowed_planning_time = 5.0

    constraints = Constraints()
    constraints.name = "closed_loop_tcp_goal"

    pos = PositionConstraint()
    pos.header.frame_id = robot.base_frame
    pos.link_name = robot.tool_frame
    pos.weight = 1.0

    region = BoundingVolume()

    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.SPHERE
    primitive.dimensions = [0.008]

    pose = Pose()
    pose.position.x = float(target[0])
    pose.position.y = float(target[1])
    pose.position.z = float(target[2])
    pose.orientation.w = 1.0

    region.primitives = [primitive]
    region.primitive_poses = [pose]
    pos.constraint_region = region

    ori = OrientationConstraint()
    ori.header.frame_id = robot.base_frame
    ori.link_name = robot.tool_frame

    ori.orientation.x = float(quat[0])
    ori.orientation.y = float(quat[1])
    ori.orientation.z = float(quat[2])
    ori.orientation.w = float(quat[3])

    ori.absolute_x_axis_tolerance = 0.10
    ori.absolute_y_axis_tolerance = 0.10
    ori.absolute_z_axis_tolerance = 0.10
    ori.weight = 1.0

    constraints.position_constraints = [pos]
    constraints.orientation_constraints = [ori]
    req.goal_constraints = [constraints]

    if not robot.motion_plan_client.wait_for_service(
        timeout_sec=5.0
    ):
        raise RuntimeError("MOTION_PLAN_SERVICE_UNAVAILABLE")

    future = robot.motion_plan_client.call_async(request)

    rclpy.spin_until_future_complete(
        robot,
        future,
        timeout_sec=10.0
    )

    response = future.result()

    if response is None:
        raise RuntimeError("NO_PLANNING_RESPONSE")

    plan = response.motion_plan_response

    if plan.error_code.val != 1:
        raise RuntimeError(
            f"PLANNING_FAILED:{plan.error_code.val}"
        )

    return plan.trajectory


rclpy.init()

robot = GazeboPandaAdapter()
runtime = RobotRuntime()

try:
    skill = GraspObject(
        object_id="cube",
        strategy="top",
        approach_height=0.08,
        lift_height=0.08,
    )

    #
    # 1. Main perception
    #
    main = RGBDSceneObserver(
        query="cube"
    )

    try:
        main_scene = main.observe_once(
            timeout=10.0
        )
    finally:
        main.destroy_node()

    if main_scene is None:
        raise RuntimeError("MAIN_OBJECT_NOT_FOUND")

    print("\n=== MAIN SCENE ===")
    print(main_scene)

    #
    # 2. Detect current workspace
    #
    workspace_observer = WorkspaceObserver()

    try:
        ws = workspace_observer.observe_once(
            timeout=5.0
        )
    finally:
        workspace_observer.destroy_node()

    if ws is None:
        raise RuntimeError("WORKSPACE_NOT_FOUND")

    workspace = {
        "x_min": ws["x_min"],
        "x_max": ws["x_max"],
        "y_min": ws["y_min"],
        "y_max": ws["y_max"],
        "z": ws["table_z"],
    }

    print("\n=== WORKSPACE ===")
    print(workspace)

    #
    # 3. Initial semantic grasp plan
    #
    initial_plan = runtime.plan_grasp_object(
        robot,
        skill
    )

    print("\n=== INITIAL GRASP PLAN ===")
    print(initial_plan)

    if not initial_plan["success"]:
        raise RuntimeError(
            initial_plan["failure_reason"]
        )

    approach_target = np.asarray(
        initial_plan["approach_target"]
    )

    approach_quat = np.asarray(
        initial_plan["approach_orientation"]
    )

    #
    # Open gripper before approaching the object
    #
    print("\nOPENING GRIPPER...")

    open_result = robot.move_gripper(0.04)

    print("\nGRIPPER OPEN RESULT:")
    print(open_result)

    if hasattr(open_result, "success") and not open_result.success:
        raise RuntimeError(
            "GRIPPER_OPEN_FAILED"
        )

    #
    # 4. MoveIt plan + execute approach
    #
    trajectory = plan_tcp(
        robot,
        approach_target,
        approach_quat
    )

    print("\nAPPROACH TRAJECTORY POINTS:")
    print(
        len(
            trajectory.joint_trajectory.points
        )
    )

    result = robot.execute_planned_trajectory(
        trajectory
    )

    print("\nAPPROACH EXECUTION:")
    print(result)

    final_tcp, _ = robot.get_tcp_pose()

    print("\nAPPROACH FINAL TCP:")
    print(final_tcp)

    print("\nAPPROACH TCP ERROR:")
    print(
        float(
            np.linalg.norm(
                approach_target - final_tcp
            )
        )
    )

    if not result.success:
        raise RuntimeError(
            "APPROACH_EXECUTION_FAILED"
        )

    #
    # 5. Side-camera handoff
    #
    verify = runtime.reobserve_grasp_object(
        robot=robot,
        skill=skill,
        previous_scene=main_scene,
        workspace=workspace,
    )

    print("\n=== VERIFY RESULT ===")
    print(verify)


    if hasattr(
        runtime,
        "last_grasp_verify_residual"
    ):
        print(
            "\nVERIFY PLANAR RESIDUAL:"
        )
        print(
            runtime.last_grasp_verify_residual
        )

    if not verify["success"]:
        raise RuntimeError(
            verify["failure_reason"]
        )

    verify_scene = verify["scene"]
    obj = verify_scene.objects[
        skill.object_id
    ]

    #
    # 6. Recompute grasp from fresh side-camera XYZ
    #
    object_center = np.asarray(
        obj.position_robot,
        dtype=float
    )

    object_center[2] = (
        float(obj.support_z)
        + float(obj.height) / 2.0
    )

    replan = robot.find_reachable_grasp_pose(
        contact_point=object_center,
        support_z=float(obj.support_z),
        approach_distance=float(
            skill.approach_height
        ),
    )

    print("\n=== VERIFY REPLANNED GRASP ===")
    print(replan)


    if not replan["success"]:
        raise RuntimeError(
            replan["failure_reason"]
        )

    #
    # 7. PLAN descend only — DO NOT EXECUTE yet
    #
    descend_trajectory = plan_tcp(
        robot,
        np.asarray(
            replan["grasp_target"]
        ),
        np.asarray(
            replan["orientation"]
        ),
    )

    print("\nDESCEND MOVEIT PLAN: SUCCESS")
    print(
        "DESCEND TRAJECTORY POINTS:",
        len(
            descend_trajectory
            .joint_trajectory
            .points
        )
    )


    #
    # Pre-descend clearance validation
    #
    cube_width = 0.040
    finger_position = 0.04

    total_opening = (
        2.0 * finger_position
    )

    required_opening = (
        cube_width
        + 0.010
    )

    print("\nPRE-DESCEND CLEARANCE CHECK:")
    print("total_opening:", total_opening)
    print("required_opening:", required_opening)

    if total_opening < required_opening:
        raise RuntimeError(
            "INSUFFICIENT_GRIPPER_CLEARANCE"
        )

    #
    # Check grasp center alignment with fresh object observation
    #
    grasp_target = np.asarray(
        replan["grasp_target"],
        dtype=float
    )

    tool_axis = np.asarray(
        replan["tool_axis"],
        dtype=float
    )

    contact_offset = float(
        replan["contact_offset"]
    )

    predicted_contact = (
        grasp_target
        + tool_axis * contact_offset
    )

    object_center = np.asarray(
        obj.position_robot,
        dtype=float
    ).copy()

    object_center[2] = (
        float(obj.support_z)
        + float(obj.height) / 2.0
    )

    lateral_error = float(
        np.linalg.norm(
            predicted_contact[:2]
            - object_center[:2]
        )
    )

    print("\nPRE-DESCEND ALIGNMENT CHECK:")
    print("object_center:", object_center)
    print("predicted_contact:", predicted_contact)
    print("lateral_error:", lateral_error)

    if lateral_error > 0.010:
        raise RuntimeError(
            "GRASP_CENTER_MISALIGNED"
        )

    print("\nEXECUTING DESCEND...")

    descend_result = robot.execute_planned_trajectory(
        descend_trajectory
    )

    print("\nDESCEND EXECUTION:")
    print(descend_result)

    descend_tcp, _ = robot.get_tcp_pose()

    grasp_target = np.asarray(
        replan["grasp_target"]
    )

    print("\nDESCEND FINAL TCP:")
    print(descend_tcp)

    descend_error = float(
        np.linalg.norm(
            grasp_target - descend_tcp
        )
    )

    print("\nDESCEND TCP ERROR:")
    print(descend_error)


    #
    # Post-descend visual verification
    #
    print("\n=== POST-DESCEND VERIFY ===")

    post_verify = runtime.reobserve_grasp_object(
        robot=robot,
        skill=skill,
        previous_scene=verify_scene,
        workspace=workspace,
    )

    print(post_verify)

    post_descend_occluded = False

    if not post_verify["success"]:
        if (
            post_verify.get("failure_reason")
            == "VERIFY_OBJECT_NOT_FOUND"
        ):
            post_descend_occluded = True

            print(
                "\nPOST-DESCEND STATUS: "
                "OBJECT_OCCLUDED_AT_GRASP"
            )

            #
            # Close gripper on the object.
            #
            print("\nCLOSING GRIPPER...")

            close_result = robot.move_gripper(
                0.0
            )

            print("\nGRIPPER CLOSE RESULT:")
            print(close_result)

            print(
                "\nGRIPPER ACTUAL POSITION:"
            )
            print(
                close_result.actual
            )

            q = np.asarray(
                close_result.actual,
                dtype=float
            )

            if q.size != 2:
                raise RuntimeError(
                    "INVALID_GRIPPER_STATE"
                )

            q1 = float(q[0])
            q2 = float(q[1])

            asymmetry = abs(q1 - q2)

            print(
                "\nGRIPPER CONTACT ASYMMETRY:"
            )
            print(asymmetry)

            if (
                q1 > 0.002
                and q2 > 0.002
                and asymmetry <= 0.006
            ):
                print(
                    "\nGRASP CONTACT STATUS: "
                    "PROVISIONAL_CONTACT"
                )

            else:
                print(
                    "\nGRASP CONTACT STATUS: "
                    "ASYMMETRIC"
                )

                #
                # Closed-loop grasp recentering.
                #
                # Finger asymmetry estimates object displacement from
                # the gripper mid-plane. Do not compensate the entire
                # estimated error in one motion: perception, contact,
                # compliance and execution all contain uncertainty.
                #
                estimated_hand_y_error = (
                    (q1 - q2) / 2.0
                )

                recenter_gain = 0.35
                max_recenter_step = 0.004

                lateral_correction = float(
                    np.clip(
                        -recenter_gain
                        * estimated_hand_y_error,
                        -max_recenter_step,
                        max_recenter_step,
                    )
                )

                print(
                    "\nESTIMATED HAND-Y ERROR:"
                )
                print(estimated_hand_y_error)

                print(
                    "\nBOUNDED RECENTER STEP:"
                )
                print(lateral_correction)

                #
                # Re-open before lateral correction.
                #
                reopen_result = robot.move_gripper(
                    0.04
                )

                print(
                    "\nGRIPPER REOPEN RESULT:"
                )
                print(reopen_result)

                if not reopen_result.success:
                    raise RuntimeError(
                        "GRIPPER_REOPEN_FAILED"
                    )

                #
                # Current grasp orientation defines the
                # gripper local-Y closing axis.
                #
                q_orient = np.asarray(
                    replan["orientation"],
                    dtype=float
                )

                x, y, z, w = q_orient

                R_hand = np.array([
                    [
                        1 - 2*(y*y + z*z),
                        2*(x*y - z*w),
                        2*(x*z + y*w)
                    ],
                    [
                        2*(x*y + z*w),
                        1 - 2*(x*x + z*z),
                        2*(y*z - x*w)
                    ],
                    [
                        2*(x*z - y*w),
                        2*(y*z + x*w),
                        1 - 2*(x*x + y*y)
                    ]
                ], dtype=float)

                hand_y_axis = R_hand[:, 1]

                correction = (
                    hand_y_axis
                    * lateral_correction
                )

                corrected_target = (
                    np.asarray(
                        replan["grasp_target"],
                        dtype=float
                    )
                    + correction
                )

                print(
                    "\nHAND-Y AXIS:"
                )
                print(hand_y_axis)

                print(
                    "\nCORRECTION VECTOR:"
                )
                print(correction)

                print(
                    "\nCORRECTED GRASP TARGET:"
                )
                print(corrected_target)

                #
                # Plan and execute lateral correction
                # while remaining at grasp height.
                #
                correction_trajectory = plan_tcp(
                    robot,
                    corrected_target,
                    np.asarray(
                        replan["orientation"]
                    ),
                )

                correction_result = (
                    robot.execute_planned_trajectory(
                        correction_trajectory
                    )
                )

                print(
                    "\nLATERAL CORRECTION EXECUTION:"
                )
                print(correction_result)

                if not correction_result.success:
                    raise RuntimeError(
                        "LATERAL_CORRECTION_FAILED"
                    )

                #
                # Close again after centering correction.
                #
                print(
                    "\nCLOSING GRIPPER AFTER CORRECTION..."
                )

                retry_close = robot.move_gripper(
                    0.0
                )

                print(
                    "\nRETRY GRIPPER CLOSE RESULT:"
                )
                print(retry_close)

                retry_q = np.asarray(
                    retry_close.actual,
                    dtype=float
                )

                retry_q1 = float(retry_q[0])
                retry_q2 = float(retry_q[1])

                retry_asymmetry = abs(
                    retry_q1 - retry_q2
                )

                print(
                    "\nRETRY CONTACT ASYMMETRY:"
                )
                print(retry_asymmetry)

                if (
                    retry_q1 > 0.002
                    and retry_q2 > 0.002
                    and retry_asymmetry <= 0.006
                ):
                    print(
                        "\nGRASP CONTACT STATUS: "
                        "PROVISIONAL_CONTACT"
                    )

                else:
                    raise RuntimeError(
                        "GRASP_RECENTER_FAILED"
                    )

        else:
            raise RuntimeError(
                "POST_DESCEND_VERIFY_FAILED"
            )

    if not post_descend_occluded:

        post_scene = post_verify["scene"]
        post_obj = post_scene.objects[
            skill.object_id
        ]

        pre_world = np.asarray(
            obj.position_world,
            dtype=float
        )

        post_world = np.asarray(
            post_obj.position_world,
            dtype=float
        )

        object_displacement = float(
            np.linalg.norm(
                post_world - pre_world
            )
        )

        print("\nPRE-DESCEND OBJECT WORLD:")
        print(pre_world)

        print("\nPOST-DESCEND OBJECT WORLD:")
        print(post_world)

        print("\nOBJECT DISPLACEMENT:")
        print(object_displacement)

        if object_displacement > 0.015:
            raise RuntimeError(
                "OBJECT_MOVED_DURING_DESCEND"
            )

    if not descend_result.success:
        raise RuntimeError(
            "DESCEND_EXECUTION_FAILED"
        )

finally:
    robot.destroy_node()
    rclpy.shutdown()
