"""Interactive semantic terminal for the Physical AI runtime."""

import argparse

import rclpy

from physical_ai_runtime.adapters.gazebo_panda_adapter import GazeboPandaAdapter
from physical_ai_runtime.ai.agent.groq_reasoner import GroqReasoner
from physical_ai_runtime.ai.agent.planner import SemanticPlanner
from physical_ai_runtime.ai.groq_task_program import GroqTaskProgramInterpreter
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

    print("Physical AI terminal")
    print("Commands: task <request> | status | observe <object> | plan grasp <object> | advise grasp <object> | grasp <object> | release | abort | quit")
    print("Execution:", "enabled" if args.allow_execution else "plan-only")

    try:
        while rclpy.ok():
            text = input("physical-ai> ").strip()
            if text in {"quit", "exit"}:
                break

            try:
                command = policy.parse(text)
                allowed, reason = policy.authorize(command)
                if not allowed:
                    print({"success": False, "reason": reason})
                    continue
                execute_command(
                    command,
                    robot=robot,
                    runtime=runtime,
                    execution_enabled=args.allow_execution,
                )
            except Exception as exc:
                print({"success": False, "reason": str(exc)})
    finally:
        robot.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
