import time

import numpy as np
import rclpy

from physical_ai_runtime.adapters.gazebo_panda_adapter import (
    GazeboPandaAdapter,
)

from physical_ai_runtime.perception.scene_state import (
    SceneObject,
    SceneState,
)

from physical_ai_runtime.planning.viewpoint_planner import (
    ViewpointPlanner,
)

from physical_ai_runtime.planning.viewpoint_plan_validator import (
    ViewpointPlanValidator,
)


C = "\033[1;36m"
G = "\033[1;32m"
R = "\033[1;31m"
Y = "\033[1;33m"
W = "\033[0m"


def wait_for_tf(
    robot,
    target_frame,
    source_frame,
    timeout_s=5.0,
):
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        rclpy.spin_once(
            robot,
            timeout_sec=0.1,
        )

        try:
            robot.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
            )

            return True

        except Exception:
            pass

    return False


def main():
    rclpy.init()

    robot = GazeboPandaAdapter()

    try:
        print(
            f"\n{C}"
            "════════ MOVEIT VIEWPOINT PLAN-ONLY ════════"
            f"{W}"
        )

        #
        # 1. TF must exist.
        #
        if not wait_for_tf(
            robot,
            robot.base_frame,
            "world",
            timeout_s=5.0,
        ):
            print(
                f"{R}"
                "✗ WORLD → PANDA BASE TF unavailable"
                f"{W}"
            )

            raise SystemExit(1)

        print(
            f"{G}"
            "✓ TF READY"
            f"{W}"
        )

        #
        # 2. PlanningScene safety context must exist
        # before any autonomous planning attempt.
        #
        safety = (
            robot.verify_motion_safety_context()
        )

        print(
            f"{Y}safety success:{W}",
            safety["success"],
        )

        print(
            f"{Y}present collision ids:{W}",
            safety.get(
                "present_collision_ids",
                [],
            ),
        )

        if not safety["success"]:
            print(
                f"{R}"
                "✗ SAFETY CONTEXT FAILED:"
                f"{W}",
                safety.get(
                    "failure_reason",
                ),
            )

            raise SystemExit(1)

        #
        # 3. Known verified cube scene.
        #
        scene = SceneState(
            objects={
                "cube": SceneObject(
                    object_id="cube",

                    position_world=np.array([
                        0.8487153011,
                        -0.3597458251,
                        0.7437657319,
                    ]),

                    position_robot=np.array([
                        0.2487153011,
                        -0.3597458251,
                        0.7437657319,
                    ]),

                    depth=0.696184814,

                    size_xyz=np.array([
                        0.03641773,
                        0.03892930,
                        0.0,
                    ]),

                    height=None,
                    support_z=None,
                )
            }
        )

        #
        # 4. Generate camera candidates.
        #
        viewpoint_planner = (
            ViewpointPlanner()
        )

        candidate_plan = (
            viewpoint_planner
            .generate_candidates(
                scene,
                object_id="cube",
                semantic_intent=(
                    "obtain an unobstructed view "
                    "of the target"
                ),
            )
        )

        assert (
            candidate_plan
            .safe_to_plan_motion
        )

        #
        # 5. PLAN ONLY.
        #
        validator = (
            ViewpointPlanValidator(
                robot
            )
        )

        selection = (
            validator
            .validate_candidates(
                candidate_plan.candidates,
                timeout=4.0,
            )
        )

        print(
            f"\n{C}"
            "════════ CANDIDATE RESULTS ════════"
            f"{W}"
        )

        for result in selection.results:

            if result.success:
                status = (
                    f"{G}PLAN_OK{W}"
                )
            else:
                status = (
                    f"{R}REJECT{W}"
                )

            print(
                f"\n{Y}"
                f"{result.candidate_id}"
                f"{W}"
            )

            print(
                "  status:",
                status,
            )

            print(
                "  geometric_score:",
                round(
                    result.geometric_score,
                    3,
                ),
            )

            print(
                "  hand_world:",
                result
                .hand_position_world
                .round(4)
                .tolist(),
            )

            print(
                "  hand_base:",
                result
                .hand_position_base
                .round(4)
                .tolist(),
            )

            print(
                "  failure_reason:",
                result.failure_reason,
            )

            if result.success:
                print(
                    "  joint_path_length:",
                    round(
                        result.joint_path_length,
                        4,
                    ),
                )

                print(
                    "  selection_score:",
                    round(
                        result.selection_score,
                        4,
                    ),
                )

            if result.success:
                points = (
                    result
                    .trajectory
                    .joint_trajectory
                    .points
                )

                print(
                    "  trajectory_points:",
                    len(points),
                )

                if points:
                    print(
                        "  trajectory_duration_s:",
                        round(
                            points[-1].time_from_start.sec
                            + points[-1].time_from_start.nanosec
                            / 1e9,
                            4,
                        ),
                    )

                    start = np.array(
                        points[0].positions,
                        dtype=float,
                    )

                    finish = np.array(
                        points[-1].positions,
                        dtype=float,
                    )

                    print(
                        "  joint_displacement_norm:",
                        round(
                            float(
                                np.linalg.norm(
                                    finish - start
                                )
                            ),
                            4,
                        ),
                    )

        print(
            f"\n{C}"
            "════════ SELECTION ════════"
            f"{W}"
        )

        print(
            "safe_candidate_available:",
            selection.safe_candidate_available,
        )

        print(
            "reason:",
            selection.reason,
        )

        if selection.selected:
            print(
                f"{G}"
                "selected:"
                f"{W}",
                selection
                .selected
                .candidate_id,
            )

            print(
                "selected_score:",
                selection
                .selected
                .geometric_score,
            )

            print(
                "selected_motion_score:",
                round(
                    selection
                    .selected
                    .selection_score,
                    4,
                ),
            )

        else:
            print(
                f"{R}"
                "selected: NONE"
                f"{W}"
            )

        print(
            f"\n{C}"
            "PLAN ONLY — "
            "NO TRAJECTORY EXECUTION — "
            "NO ROBOT MOTION"
            f"{W}"
        )

    finally:
        robot.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
