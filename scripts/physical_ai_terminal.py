"""Interactive semantic terminal for the Physical AI runtime."""

import argparse

import rclpy

from physical_ai_runtime.adapters.gazebo_panda_adapter import GazeboPandaAdapter
from physical_ai_runtime.ai.agent.groq_reasoner import GroqReasoner
from physical_ai_runtime.ai.agent.planner import SemanticPlanner
from physical_ai_runtime.ai.groq_task_program import GroqTaskProgramInterpreter
from physical_ai_runtime.ai.conversation_agent import PhysicalAIConversationAgent
from physical_ai_runtime.ai.semantic_terminal import SemanticTerminalPolicy
from physical_ai_runtime.core.feedback import SemanticFeedback
from physical_ai_runtime.core.runtime import RobotRuntime
from physical_ai_runtime.core.skills import GraspObject, Release
from physical_ai_runtime.perception.camera_manager import CameraManager


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-execution",
        action="store_true",
        help="Permit semantic grasp/release commands after runtime safety checks.",
    )
    return parser


def execute_command(command, *, robot, runtime, execution_enabled=False):
    if command.name == "status":
        position, orientation = robot.get_tcp_pose()
        print({"tcp_position": position.tolist(), "tcp_orientation": orientation.tolist()})
        return

    if command.name == "observe":
        manager = CameraManager(query=command.object_id)
        try:
            result = manager.observe_best(timeout_per_camera=5.0)
        finally:
            manager.close()
        print({"camera": result.camera, "status": result.status, "scene": result.scene})
        return

    if command.name == "plan_grasp":
        plan = runtime.plan_grasp_object(robot, GraspObject(command.object_id))
        print(plan)
        return

    if command.name == "advise_grasp":
        plan = runtime.plan_grasp_object(robot, GraspObject(command.object_id))
        feedback = SemanticFeedback(
            state=("GRASP_READY" if plan.get("success") else "REPLAN_REQUIRED"),
            object_id=command.object_id,
            message="Fresh runtime grasp-plan result; no execution has occurred.",
            confidence=1.0 if plan.get("success") else 0.0,
            metrics={
                "plan_success": bool(plan.get("success")),
                "strategy": plan.get("strategy"),
                "failure_code": plan.get("code"),
            },
            available_actions=["REOBSERVE", "ABORT"],
        )
        recommendation = SemanticPlanner(GroqReasoner()).choose_action(feedback)
        print({"plan": plan, "recommendation": recommendation})
        return

    if command.name == "task_program":
        program = GroqTaskProgramInterpreter().interpret(command.prompt)
        result = runtime.execute_semantic_program(
            robot,
            program,
            execute=execution_enabled,
        )
        print({"program": program, "result": result})
        return

    if command.name == "grasp":
        feedback = runtime.execute_grasp_attempt(
            robot,
            GraspObject(command.object_id),
        )
        print(feedback)
        return

    if command.name == "release":
        print(runtime.execute(robot, Release()))
        return

    if command.name == "abort":
        print(robot.stop())
        return

    raise RuntimeError("TERMINAL_COMMAND_UNHANDLED")


def main():
    args = build_parser().parse_args()
    policy = SemanticTerminalPolicy(execution_enabled=args.allow_execution)

    rclpy.init()
    robot = GazeboPandaAdapter()
    runtime = RobotRuntime()
    conversation = PhysicalAIConversationAgent()

    print("Physical AI conversational terminal")
    print("Talk naturally. Hard commands: abort | quit")
    print("Legacy commands remain available for debugging.")
    print("Execution:", "enabled" if args.allow_execution else "plan-only")

    try:
        while rclpy.ok():
            text = input("physical-ai> ").strip()

            if text.lower() in {"quit", "exit"}:
                break

            try:
                # ABORT is intentionally hard-wired and never routed
                # through the language model.
                if text.lower() == "abort":
                    print(robot.stop())
                    continue

                # Preserve explicit legacy/debug commands.
                legacy_prefixes = (
                    "status",
                    "observe ",
                    "plan grasp ",
                    "advise grasp ",
                    "grasp ",
                    "release",
                    "task ",
                )

                if any(
                    text.lower() == prefix
                    or text.lower().startswith(prefix)
                    for prefix in legacy_prefixes
                ):
                    try:
                        command = policy.parse(text)
                    except ValueError:
                        command = None

                    if command is not None:
                        allowed, reason = policy.authorize(command)

                        if not allowed:
                            print(
                                {
                                    "success": False,
                                    "reason": reason,
                                }
                            )
                            continue

                        execute_command(
                            command,
                            robot=robot,
                            runtime=runtime,
                            execution_enabled=args.allow_execution,
                        )
                        continue

                lowered = text.lower()

                vision_terms = (
                    "เห็นอะไร",
                    "มองเห็นอะไร",
                    "เห็นไหม",
                    "เห็น cube",
                    "เห็นคิวบ์",
                    "กล้องเห็น",
                    "บนโต๊ะมีอะไร",
                    "what do you see",
                    "can you see",
                    "do you see",
                )

                if any(term in lowered for term in vision_terms):
                    queries = [
                        "cube",
                        "cylinder",
                        "sphere",
                    ]

                    visible = []
                    diagnostics = []

                    for query in queries:
                        manager = CameraManager(query=query)

                        try:
                            result = manager.observe_verified(
                                query=query,
                                task=(
                                    f"Verify whether {query} is visible "
                                    "in the current scene."
                                ),
                                timeout=5.0,
                            )
                        finally:
                            manager.close()

                        scene = result.get("scene")
                        best_camera = result.get("best_camera")
                        safe = bool(
                            result.get("safe_visual_evidence", False)
                        )
                        reason = result.get("reason")

                        diagnostics.append(
                            {
                                "object": query,
                                "camera": best_camera,
                                "safe_visual_evidence": safe,
                                "reason": reason,
                            }
                        )

                        if (
                            safe
                            and scene is not None
                            and query in scene.objects
                        ):
                            obj = scene.objects[query]

                            position = (
                                None
                                if obj.position_world is None
                                else [
                                    float(v)
                                    for v in obj.position_world
                                ]
                            )

                            visible.append(
                                {
                                    "object": query,
                                    "camera": best_camera,
                                    "position_world": position,
                                }
                            )

                    if visible:
                        names = ", ".join(
                            item["object"]
                            for item in visible
                        )

                        print(
                            "AI> ตอนนี้จาก perception ของระบบ "
                            f"ผมเห็น {names}"
                        )

                        for item in visible:
                            print(
                                "   ",
                                item["object"],
                                "camera=",
                                item["camera"],
                                "world=",
                                item["position_world"],
                            )
                    else:
                        print(
                            "AI> ตอนนี้ perception ยังยืนยัน "
                            "วัตถุที่ตรวจสอบไม่ได้"
                        )
                        print(
                            {
                                "perception": diagnostics
                            }
                        )

                    continue

                turn = conversation.respond(text)

                print("AI>", turn.message)

                if turn.mode != "task":
                    continue

                result = runtime.execute_semantic_program(
                    robot,
                    turn.program,
                    execute=args.allow_execution,
                )

                print(
                    {
                        "mode": "task",
                        "execution": (
                            "enabled"
                            if args.allow_execution
                            else "plan-only"
                        ),
                        "result": result,
                    }
                )

            except Exception as exc:
                print(
                    {
                        "success": False,
                        "reason": str(exc),
                    }
                )
    finally:
        robot.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
