import time

import numpy as np
import rclpy

from physical_ai_runtime.adapters.gazebo_panda_adapter import (
    GazeboPandaAdapter,
)

from physical_ai_runtime.planning.execution_authorization import (
    ExecutionAuthorizationGate,
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


def main():
    rclpy.init()

    robot = GazeboPandaAdapter()

    try:
        #
        # TF listener must be warm before viewpoint planning.
        #
        deadline = time.time() + 5.0
        tf_ready = False

        while time.time() < deadline:
            rclpy.spin_once(
                robot,
                timeout_sec=0.1,
            )

            try:
                robot.tf_buffer.lookup_transform(
                    robot.base_frame,
                    "world",
                    rclpy.time.Time(),
                )

                tf_ready = True
                break

            except Exception:
                pass

        if not tf_ready:
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

        planner = ViewpointPlanner()

        candidate_plan = (
            planner.generate_candidates(
                scene,
                object_id="cube",
            )
        )

        validator = (
            ViewpointPlanValidator(
                robot
            )
        )

        selection = (
            validator.validate_candidates(
                candidate_plan.candidates,
                timeout=4.0,
            )
        )

        if not selection.safe_candidate_available:
            print(
                f"{R}"
                "✗ NO SAFE PLAN"
                f"{W}"
            )

            for result in selection.results:
                print(
                    result.candidate_id,
                    "->",
                    result.failure_reason,
                )

            raise SystemExit(1)

        selected = selection.selected

        planned_at = time.monotonic()

        gate = ExecutionAuthorizationGate(
            robot,
            max_start_state_error=0.12,
            max_trajectory_age_s=2.0,
        )

        result = gate.authorize(
            selected.trajectory,
            planned_at_monotonic=(
                planned_at
            ),
        )

        print(
            f"\n{C}"
            "════════ EXECUTION AUTHORIZATION ════════"
            f"{W}"
        )

        print(
            f"{Y}candidate:{W}",
            selected.candidate_id,
        )

        print(
            f"{Y}authorized:{W}",
            result.authorized,
        )

        print(
            f"{Y}reason:{W}",
            result.reason,
        )

        print(
            f"{Y}start_state_error:{W}",
            result.start_state_error,
        )

        print(
            f"{Y}trajectory_age_s:{W}",
            result.trajectory_age_s,
        )

        print(
            f"{Y}safety_success:{W}",
            result
            .safety_context
            .get(
                "success",
                False,
            ),
        )

        verified, verify_reason = (
            gate.verify_authorization(
                selected.trajectory,
                result,
            )
        )

        print(
            f"{Y}trajectory_bound:{W}",
            verified,
        )

        print(
            f"{Y}verify_reason:{W}",
            verify_reason,
        )

        assert verified

        if result.authorized:
            print(
                f"\n{G}"
                "✓ EXECUTION WOULD BE AUTHORIZED"
                f"{W}"
            )
        else:
            print(
                f"\n{R}"
                "✗ EXECUTION DENIED"
                f"{W}"
            )

        print(
            f"\n{C}"
            "AUTHORIZATION CHECK ONLY — "
            "NO TRAJECTORY EXECUTION"
            f"{W}"
        )

    finally:
        robot.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
