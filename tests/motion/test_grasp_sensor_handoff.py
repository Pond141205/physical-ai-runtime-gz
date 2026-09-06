import rclpy

from physical_ai_runtime.adapters.gazebo_panda_adapter import GazeboPandaAdapter
from physical_ai_runtime.core.runtime import RobotRuntime
from physical_ai_runtime.core.skills import GraspObject


WORKSPACE = {
    "x_min": 0.48135619,
    "x_max": 0.95986138,
    "y_min": -0.68864381,
    "y_max": -0.21013862,
    "z": 0.7250000,
}


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

    print("\n=== INITIAL MAIN-CAMERA PLAN ===")

    initial_plan = runtime.plan_grasp_object(
        robot,
        skill
    )

    print(initial_plan)

    if not initial_plan.get("success"):
        raise RuntimeError(
            initial_plan.get(
                "failure_reason",
                "INITIAL_PLAN_FAILED"
            )
        )

    #
    # Rebuild the previous scene so verify observer has the
    # main-camera spatial prior + object geometry.
    #
    from rgbd_scene_observer import RGBDSceneObserver

    main_observer = RGBDSceneObserver(
        query=skill.object_id
    )

    try:
        previous_scene = main_observer.observe_once(
            timeout=10.0
        )
    finally:
        main_observer.destroy_node()

    if previous_scene is None:
        raise RuntimeError(
            "MAIN_SCENE_NOT_AVAILABLE"
        )

    print("\n=== MAIN SCENE ===")
    print(previous_scene)

    print("\n=== VERIFY CAMERA REOBSERVE ===")

    verify_result = runtime.reobserve_grasp_object(
        robot=robot,
        skill=skill,
        previous_scene=previous_scene,
        workspace=WORKSPACE,
    )

    print(verify_result)

    if not verify_result.get("success"):
        raise RuntimeError(
            verify_result.get(
                "failure_reason",
                "VERIFY_FAILED"
            )
        )

    verify_scene = verify_result["scene"]
    verify_obj = verify_scene.objects[
        skill.object_id
    ]

    print("\nVERIFY WORLD:")
    print(verify_obj.position_world)

    print("\nVERIFY ROBOT:")
    print(verify_obj.position_robot)

    print("\nVERIFY HEIGHT:")
    print(verify_obj.height)

    print("\nVERIFY SUPPORT Z:")
    print(verify_obj.support_z)

finally:
    robot.destroy_node()
    rclpy.shutdown()
