"""Interactive semantic terminal for the Physical AI runtime."""

import argparse
import os
import warnings

# Keep third-party model initialization out of the
# user-facing Physical AI terminal.
os.environ.setdefault(
    "HF_HUB_DISABLE_PROGRESS_BARS",
    "1",
)
os.environ.setdefault(
    "TRANSFORMERS_NO_ADVISORY_WARNINGS",
    "1",
)
os.environ.setdefault(
    "TOKENIZERS_PARALLELISM",
    "false",
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="transformers",
)


import rclpy

from physical_ai_runtime.adapters.gazebo_panda_adapter import GazeboPandaAdapter
from physical_ai_runtime.ai.groq_task_program import GroqTaskProgramInterpreter
from physical_ai_runtime.ai.conversation_agent import PhysicalAIConversationAgent
from physical_ai_runtime.ai.semantic_terminal import SemanticTerminalPolicy
from physical_ai_runtime.ai.semantic_manipulation_task import (
    SemanticTaskProgram,
)
from physical_ai_runtime.core.runtime import RobotRuntime
from physical_ai_runtime.core.status import RuntimeStatus
from physical_ai_runtime.perception.camera_manager import CameraManager


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-execution",
        action="store_true",
        help="Permit validated primitive commands after runtime safety checks.",
    )
    parser.add_argument(
        "--continuous-query",
        default=None,
        help="Start background detection/tracking for this semantic label.",
    )
    parser.add_argument(
        "--continuous-rate-hz",
        type=float,
        default=2.0,
        help="Background detector frequency.",
    )
    return parser


def execute_command(
    command,
    *,
    robot,
    runtime,
    execution_enabled=False,
    perception_manager=None,
):
    if command.name in {"move_to", "open", "close", "stop"}:
        if command.name == "stop":
            result = robot.stop()
            print(result)
            return result

        step = {"action": command.name.upper()}
        if command.name == "move_to":
            step.update({
                "frame_id": command.frame_id,
                "position": list(command.position),
            })
            if command.orientation is not None:
                step["orientation"] = list(command.orientation)

        program = SemanticTaskProgram.from_model_output({
            "steps": [step],
        })
        result = runtime.execute_semantic_program(
            robot,
            program,
            execute=execution_enabled,
            perception_manager=perception_manager,
        )
        print(result)
        return result

    if command.name == "status":
        position, orientation = robot.get_tcp_pose()
        perception = None
        if perception_manager is not None:
            frame = (
                perception_manager
                .get_continuous_perception_status()
            )
            if frame is not None:
                perception = {
                    "query": frame.query,
                    "valid": frame.valid,
                    "reason": frame.reason,
                    "observation_timestamp":
                        frame.observation_timestamp,
                    "detector_latency_s":
                        frame.detector_latency_s,
                    "cameras": {
                        camera: [
                            {
                                "track_id": track.track_id,
                                "label": track.label,
                                "confidence": track.confidence,
                                "state": track.state,
                                "missed": track.missed,
                            }
                            for track in tracks
                        ]
                        for camera, tracks in frame.cameras.items()
                    },
                }
        print({
            "tcp_position": position.tolist(),
            "tcp_orientation": orientation.tolist(),
            "continuous_perception": perception,
        })
        return

    if command.name == "observe":
        if perception_manager is not None:
            observation = (
                perception_manager.observe_manipulation_target(
                    query=command.object_id,
                    timeout=5.0,
                )
            )
            print({
                "camera": observation.get("camera"),
                "status": observation.get("reason"),
                "scene": observation.get("scene"),
            })
            return

        manager = CameraManager(query=command.object_id)
        try:
            result = manager.observe_best(timeout_per_camera=5.0)
        finally:
            manager.close()
        print({"camera": result.camera, "status": result.status, "scene": result.scene})
        return

    if command.name == "task_program":
        program = GroqTaskProgramInterpreter().interpret(command.prompt)
        result = runtime.execute_semantic_program(
            robot,
            program,
            execute=execution_enabled,
            perception_manager=perception_manager,
        )
        print({"program": program, "result": result})
        return result

    raise RuntimeError("TERMINAL_COMMAND_UNHANDLED")


def main():
    args = build_parser().parse_args()
    policy = SemanticTerminalPolicy(execution_enabled=args.allow_execution)

    rclpy.init()
    robot = GazeboPandaAdapter()
    runtime = RobotRuntime()
    conversation = PhysicalAIConversationAgent()

    # One persistent perception owner for the entire terminal
    # session. Sensor subscriptions, TF buffers, detector and VLM
    # resources are reused instead of recreated per question.
    camera_manager = CameraManager(
        query=args.continuous_query or "scene"
    )
    camera_manager.start_sensor_streams()
    if args.continuous_query:
        camera_manager.start_continuous_perception(
            query=args.continuous_query,
            rate_hz=args.continuous_rate_hz,
        )
        RuntimeStatus.info(
            "Continuous perception started for: "
            + args.continuous_query
        )

    print("Physical AI conversational terminal")
    print("Talk naturally. Primitives: move to | open | close | stop")
    print("Execution:", "enabled" if args.allow_execution else "plan-only")

    try:
        while rclpy.ok():
            text = input("physical-ai> ").strip()

            if text.lower() in {"quit", "exit"}:
                break

            try:
                # ABORT is intentionally hard-wired and never routed
                # through the language model.
                if text.lower() in {"stop", "abort"}:
                    stop_result = robot.stop()
                    print(stop_result)
                    tcp_position, tcp_orientation = robot.get_tcp_pose()
                    conversation.record_runtime_feedback({
                        "result_success": bool(stop_result.get("success", False)),
                        "failure_reason": stop_result.get("failure_reason"),
                        "steps": ["STOP"],
                        "tcp_position": tcp_position.tolist(),
                        "tcp_orientation": tcp_orientation.tolist(),
                        "result": str(stop_result),
                    })
                    continue

                command_prefixes = (
                    "status",
                    "observe ",
                    "move to ",
                    "open",
                    "close",
                    "stop",
                    "task ",
                )

                if any(
                    text.lower() == prefix
                    or text.lower().startswith(prefix)
                    for prefix in command_prefixes
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

                        command_result = execute_command(
                            command,
                            robot=robot,
                            runtime=runtime,
                            execution_enabled=args.allow_execution,
                            perception_manager=camera_manager,
                        )
                        if command_result is not None:
                            tcp_position, tcp_orientation = (
                                robot.get_tcp_pose()
                            )
                            conversation.record_runtime_feedback({
                                "result_success": bool(
                                    getattr(
                                        command_result,
                                        "success",
                                        False,
                                    )
                                    if not isinstance(
                                        command_result,
                                        dict,
                                    )
                                    else command_result.get(
                                        "success",
                                        False,
                                    )
                                ),
                                "failure_reason": (
                                    command_result.get(
                                        "failure_reason"
                                    )
                                    if isinstance(command_result, dict)
                                    else getattr(
                                        command_result,
                                        "failure_reason",
                                        None,
                                    )
                                ),
                                "steps": [command.name.upper()],
                                "tcp_position":
                                    tcp_position.tolist(),
                                "tcp_orientation":
                                    tcp_orientation.tolist(),
                                "result": str(command_result),
                            })
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

                if any(
                    term in lowered
                    for term in vision_terms
                ):
                    RuntimeStatus.vision(
                        "Capturing live main / side / wrist views..."
                    )

                    visual = (
                        camera_manager
                        .observe_visual_scene(
                            timeout=2.0,
                        )
                    )

                    if visual.get("success"):
                        RuntimeStatus.ok(
                            "Live visual scene available"
                        )
                    else:
                        RuntimeStatus.fail(
                            "Live visual scene unavailable: "
                            + str(
                                visual.get(
                                    "reason",
                                    "UNKNOWN",
                                )
                            )
                        )

                    context = {
                        "source":
                            "live_multiview_camera_images",
                        "verified_for_motion": False,
                        "scene_description":
                            visual.get(
                                "scene_description",
                                "",
                            ),
                        "visual_candidates":
                            visual.get(
                                "candidates",
                                [],
                            ),
                    }

                    turn = conversation.respond(
                        text,
                        runtime_context=context,
                    )

                    print("AI>", turn.message)
                    continue

                RuntimeStatus.ai(
                    "Understanding request..."
                )

                turn = conversation.respond(text)

                if turn.mode == "chat":
                    RuntimeStatus.ok(
                        "Conversation response ready"
                    )
                else:
                    RuntimeStatus.ok(
                        "Semantic robot task resolved"
                    )

                print("AI>", turn.message)

                if turn.mode != "task":
                    continue

                actions = [
                    step.action
                    for step in turn.program.steps
                ]

                RuntimeStatus.info(
                    "Task program: "
                    + " -> ".join(actions)
                )

                if args.allow_execution:
                    RuntimeStatus.safety(
                        "Execution requested; runtime safety "
                        "checks remain authoritative"
                    )
                else:
                    RuntimeStatus.info(
                        "Plan-only mode — robot will not move"
                    )

                result = runtime.execute_semantic_program(
                    robot,
                    turn.program,
                    execute=args.allow_execution,
                    perception_manager=camera_manager,
                )

                if getattr(
                    result,
                    "success",
                    False,
                ):
                    RuntimeStatus.ok(
                        "Task processing completed successfully"
                    )

                else:
                    reason = getattr(
                        result,
                        "failure_reason",
                        "UNKNOWN",
                    )

                    RuntimeStatus.fail(
                        "Task processing failed: "
                        + str(reason)
                    )

                tcp_position, tcp_orientation = robot.get_tcp_pose()
                conversation.record_runtime_feedback({
                    "result_success": bool(
                        getattr(result, "success", False)
                    ),
                    "failure_reason": getattr(
                        result,
                        "failure_reason",
                        None,
                    ),
                    "steps": actions,
                    "tcp_position": tcp_position.tolist(),
                    "tcp_orientation": tcp_orientation.tolist(),
                    "result": str(result),
                })

            except Exception as exc:
                conversation.record_runtime_feedback({
                    "result_success": False,
                    "failure_reason": str(exc),
                })
                RuntimeStatus.fail(
                    str(exc)
                )

                print(
                    {
                        "success": False,
                        "reason": str(exc),
                    }
                )
    finally:
        try:
            camera_manager.close()
        except Exception:
            pass

        robot.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
