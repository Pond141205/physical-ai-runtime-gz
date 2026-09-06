import rclpy

from physical_ai_runtime.adapters.gazebo_panda_adapter import GazeboPandaAdapter
from physical_ai_runtime.manipulation.grasp_evaluator import GraspEvaluator
from physical_ai_runtime.ai.agent.planner import SemanticPlanner
from physical_ai_runtime.ai.agent.groq_reasoner import GroqReasoner
from physical_ai_runtime.manipulation.recovery_executor import RecoveryExecutor
from physical_ai_runtime.manipulation.grasp_recovery_controller import GraspRecoveryController
from physical_ai_runtime.core.feedback import SemanticFeedback
from physical_ai_runtime.perception.rgbd_scene_observer import RGBDSceneObserver
from physical_ai_runtime.core.runtime import RobotRuntime
from physical_ai_runtime.core.skills import GraspObject


def main():
    rclpy.init()

    robot = GazeboPandaAdapter()
    observer = RGBDSceneObserver()
    runtime = RobotRuntime()

    evaluator = GraspEvaluator()

    planner = SemanticPlanner(
        reasoner=GroqReasoner(
            model="openai/gpt-oss-120b"
        )
    )

    executor = RecoveryExecutor(
        robot
    )

    controller = GraspRecoveryController(
        robot=robot,
        evaluator=evaluator,
    )

    try:
        print("\n=== AUTONOMOUS PREFLIGHT ===")

        preflight = robot.preflight_check()

        print(preflight)

        if not preflight["success"]:
            raise RuntimeError(
                "AUTONOMOUS_PREFLIGHT_FAILED:"
                + str(preflight)
            )

        print("\nPREFLIGHT PASSED")

        feedback = SemanticFeedback(
            state="READY_REACHED",
            object_id="cube",
            message=(
                "Robot is ready for a new autonomous grasp attempt."
            ),
            available_actions=[
                "REOBSERVE",
                "RETRY_GRASP",
                "ABORT",
            ],
        )

        recovery_limits = (
            robot.get_grasp_recovery_limits()
        )

        max_attempts = int(
            recovery_limits[
                "max_recenter_attempts"
            ]
        )

        attempt = 0
        previous_asymmetry = None

        semantic_step = 0
        max_semantic_steps = 12

        active_grasp_skill = None

        while True:
            semantic_step += 1

            if semantic_step > max_semantic_steps:
                print(
                    "\nSEMANTIC LOOP STOPPED: "
                    "MAX_STEPS_REACHED"
                )
                break
            print("\n==============================")
            print("SEMANTIC FEEDBACK:")
            print(feedback)

            action = planner.choose_action(
                feedback
            )

            print("\nAI ACTION:")
            print(action)

            request = executor.execute(
                action=action,
                feedback=feedback,
                grasp_context={},
            )

            print("\nRUNTIME REQUEST:")
            print(request)

            state = request.get("state")

            if state == "RECENTER_REQUESTED":
                attempt += 1

                if attempt > max_attempts:
                    print(
                        "\nRECOVERY STOPPED: "
                        "MAX_ATTEMPTS_REACHED"
                    )
                    break

                current_asymmetry = None

                if feedback.metrics is not None:
                    current_asymmetry = (
                        feedback.metrics.get(
                            "asymmetry"
                        )
                    )

                print(
                    "\nRECOVERY ATTEMPT:",
                    attempt,
                    "/",
                    max_attempts
                )

                feedback = controller.execute_recenter(
                    request
                )

                new_asymmetry = None

                if feedback.metrics is not None:
                    new_asymmetry = (
                        feedback.metrics.get(
                            "asymmetry"
                        )
                    )

                if (
                    current_asymmetry is not None
                    and new_asymmetry is not None
                ):
                    print(
                        "\nRECOVERY PROGRESS:"
                    )
                    print(
                        "before:",
                        current_asymmetry
                    )
                    print(
                        "after:",
                        new_asymmetry
                    )

                    previous_asymmetry = (
                        current_asymmetry
                    )

                continue

            if state == "RETRY_GRASP_REQUESTED":
                print("\nEXECUTING SAFE GRASP ATTEMPT...")

                strategy = (
                    action.strategy
                    if action.strategy
                    else "top"
                )

                grasp_skill = GraspObject(
                    object_id=feedback.object_id,
                    strategy=strategy,
                )

                active_grasp_skill = grasp_skill

                feedback = runtime.execute_grasp_attempt(
                    robot=robot,
                    skill=grasp_skill,
                )

                print("\nGRASP ATTEMPT FEEDBACK:")
                print(feedback)

                continue

            if state == "VERIFY_GRASP_REQUESTED":
                print(
                    "\nVERIFYING HELD OBJECT..."
                )

                if active_grasp_skill is None:
                    feedback = SemanticFeedback(
                        state="GRASP_VERIFY_FAILED",
                        object_id=feedback.object_id,
                        message=(
                            "No active grasp context "
                            "is available."
                        ),
                        available_actions=[
                            "ABORT",
                        ],
                    )
                else:
                    feedback = runtime.verify_active_grasp(
                        robot=robot,
                        skill=active_grasp_skill,
                    )

                print(
                    "\nGRASP VERIFICATION FEEDBACK:"
                )
                print(feedback)

                if feedback.state == "GRASP_VERIFIED":
                    print(
                        "\nAUTONOMOUS GRASP SUCCESS: "
                        "OBJECT SECURELY LIFTED AND VERIFIED"
                    )
                    break

                continue

            if state == "LIFT_OBJECT_REQUESTED":
                print(
                    "\nEXECUTING COLLISION-AWARE LIFT..."
                )

                if active_grasp_skill is None:
                    feedback = SemanticFeedback(
                        state="LIFT_FAILED",
                        object_id=feedback.object_id,
                        message=(
                            "No active successful grasp "
                            "context is available for lift."
                        ),
                        available_actions=[
                            "REOBSERVE",
                            "RETURN_READY",
                            "ABORT",
                        ],
                    )

                    print(
                        "\nLIFT FEEDBACK:"
                    )
                    print(feedback)

                    continue

                feedback = runtime.execute_lift(
                    robot=robot,
                    skill=active_grasp_skill,
                )

                print(
                    "\nLIFT FEEDBACK:"
                )
                print(feedback)

                continue

            if state == "RETURN_READY_REQUESTED":
                print("\nRETURNING TO READY...")

                ready_result = robot.return_to_ready()

                print("\nRETURN READY RESULT:")
                print(ready_result)

                if ready_result.success:
                    feedback = SemanticFeedback(
                        state="READY_REACHED",
                        object_id=feedback.object_id,
                        message=(
                            "Robot safely returned to its "
                            "configured ready posture."
                        ),
                        metrics={
                            "joint_error": float(
                                ready_result.error
                            )
                        },
                        available_actions=[
                            "REOBSERVE",
                            "RETRY_GRASP",
                            "CHANGE_GRASP_STRATEGY",
                            "ABORT",
                        ],
                    )
                else:
                    feedback = SemanticFeedback(
                        state="READY_FAILED",
                        object_id=feedback.object_id,
                        message=(
                            "Safe return-to-ready motion failed."
                        ),
                        metrics={
                            "failure_reason":
                                ready_result.failure_reason,
                            "joint_error": float(
                                ready_result.error
                            ),
                        },
                        available_actions=[
                            "REOBSERVE",
                            "RETURN_READY",
                            "ABORT",
                        ],
                    )

                continue

            if state == "REOBSERVE_REQUESTED":
                print("\nREOBSERVING SCENE...")

                scene = observer.observe_once(
                    timeout=10.0
                )

                object_id = feedback.object_id

                if (
                    scene is not None
                    and object_id is not None
                    and object_id in scene.objects
                ):
                    obj = scene.objects[object_id]

                    feedback = SemanticFeedback(
                        state="OBJECT_VISIBLE",
                        object_id=object_id,
                        message=(
                            "Object was re-observed "
                            "from the RGB-D scene."
                        ),
                        object_pose=(
                            obj.position_robot.copy()
                        ),
                        metrics={
                            "position_world":
                                obj.position_world.tolist(),
                            "depth":
                                float(obj.depth),
                            "height":
                                (
                                    None
                                    if obj.height is None
                                    else float(obj.height)
                                ),
                            "support_z":
                                (
                                    None
                                    if obj.support_z is None
                                    else float(obj.support_z)
                                ),
                        },
                        available_actions=[
                            "RETRY_GRASP",
                            "CHANGE_GRASP_STRATEGY",
                            "RETURN_READY",
                            "REOBSERVE",
                            "ABORT",
                        ],
                    )

                else:
                    feedback = SemanticFeedback(
                        state="OBJECT_LOST",
                        object_id=object_id,
                        message=(
                            "Object could not be found "
                            "during re-observation."
                        ),
                        available_actions=[
                            "REOBSERVE",
                            "RETURN_READY",
                            "ABORT",
                        ],
                    )

                print("\nNEW SEMANTIC FEEDBACK:")
                print(feedback)

                continue

            if state == "NEW_GRASP_STRATEGY_REQUESTED":
                print(
                    "\nEXECUTING NEW GRASP STRATEGY..."
                )

                strategy = (
                    action.strategy
                    if action.strategy
                    else None
                )

                if not strategy:
                    feedback = SemanticFeedback(
                        state="GRASP_STRATEGY_INVALID",
                        object_id=feedback.object_id,
                        message=(
                            "Planner requested a grasp strategy "
                            "change without specifying a strategy."
                        ),
                        available_actions=[
                            "REOBSERVE",
                            "CHANGE_GRASP_STRATEGY",
                            "RETURN_READY",
                            "ABORT",
                        ],
                    )

                    print(
                        "\nNEW STRATEGY FEEDBACK:"
                    )
                    print(feedback)

                    continue

                grasp_skill = GraspObject(
                    object_id=feedback.object_id,
                    strategy=strategy,
                )

                active_grasp_skill = grasp_skill

                feedback = runtime.execute_grasp_attempt(
                    robot=robot,
                    skill=grasp_skill,
                )

                print(
                    "\nNEW STRATEGY GRASP FEEDBACK:"
                )
                print(feedback)

                continue

            if state == "ABORTED":
                print("\nAI ABORTED RECOVERY")
                break

            print(
                "\nUnsupported runtime state:",
                state
            )
            break

    finally:
        observer.destroy_node()
        robot.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
