import time

import numpy as np
import rclpy

from gazebo_panda_adapter import GazeboPandaAdapter
from execution_authorization import ExecutionAuthorizationGate
from scene_state import SceneObject, SceneState
from viewpoint_planner import ViewpointPlanner
from viewpoint_plan_validator import ViewpointPlanValidator


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
            return robot.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
            )
        except Exception:
            pass

    return None


def translation(tf):
    t = tf.transform.translation

    return np.array(
        [t.x, t.y, t.z],
        dtype=float,
    )


def main():
    rclpy.init()

    robot = GazeboPandaAdapter()

    try:
        print(
            f"\n{C}"
            "════════ ACTIVE VIEW EXECUTION ════════"
            f"{W}"
        )

        # --------------------------------------------------
        # 1. TF readiness
        # --------------------------------------------------

        tf = wait_for_tf(
            robot,
            robot.base_frame,
            "world",
            timeout_s=5.0,
        )

        if tf is None:
            print(
                f"{R}✗ BASE TF UNAVAILABLE{W}"
            )
            raise SystemExit(1)

        print(
            f"{G}✓ TF READY{W}"
        )

        # --------------------------------------------------
        # 2. Known verified scene
        # --------------------------------------------------

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

        # --------------------------------------------------
        # 3. Generate viewpoint candidates
        # --------------------------------------------------

        planner = ViewpointPlanner()

        candidate_plan = planner.generate_candidates(
            scene,
            object_id="cube",
            semantic_intent=(
                "obtain an unobstructed view "
                "of the target"
            ),
        )

        if not candidate_plan.safe_to_plan_motion:
            print(
                f"{R}"
                "✗ NO GEOMETRIC VIEWPOINT"
                f"{W}"
            )
            raise SystemExit(1)

        # --------------------------------------------------
        # 4. Collision-aware PLAN ONLY
        # --------------------------------------------------

        validator = ViewpointPlanValidator(
            robot
        )

        selection = validator.validate_candidates(
            candidate_plan.candidates,
            timeout=4.0,
        )

        if not selection.safe_candidate_available:
            print(
                f"{R}"
                "✗ NO MOVEIT-SAFE VIEWPOINT"
                f"{W}"
            )

            for result in selection.results:
                print(
                    result.candidate_id,
                    "→",
                    result.failure_reason,
                )

            raise SystemExit(1)

        selected = selection.selected

        print(
            f"{Y}selected:{W}",
            selected.candidate_id,
        )

        print(
            f"{Y}selection_score:{W}",
            round(
                selected.selection_score,
                4,
            ),
        )

        # Find original camera candidate.
        original_candidate = next(
            c
            for c in candidate_plan.candidates
            if (
                c.candidate_id
                == selected.candidate_id
            )
        )

        desired_camera_world = (
            original_candidate
            .camera_position_world
        )

        print(
            f"{Y}desired camera world:{W}",
            desired_camera_world
            .round(4)
            .tolist(),
        )

        # --------------------------------------------------
        # 5. Final authorization
        # --------------------------------------------------

        planned_at = time.monotonic()

        gate = ExecutionAuthorizationGate(
            robot,
            max_start_state_error=0.12,
            max_trajectory_age_s=2.0,
        )

        authorization = gate.authorize(
            selected.trajectory,
            planned_at_monotonic=planned_at,
        )

        print(
            f"{Y}authorized:{W}",
            authorization.authorized,
        )

        print(
            f"{Y}authorization reason:{W}",
            authorization.reason,
        )

        if not authorization.authorized:
            print(
                f"{R}"
                "✗ EXECUTION DENIED"
                f"{W}"
            )
            raise SystemExit(1)

        # --------------------------------------------------
        # 6. Bind exact trajectory immediately before execute
        # --------------------------------------------------

        verified, verify_reason = (
            gate.verify_authorization(
                selected.trajectory,
                authorization,
                max_authorization_age_s=0.5,
            )
        )

        print(
            f"{Y}trajectory bound:{W}",
            verified,
        )

        print(
            f"{Y}verify reason:{W}",
            verify_reason,
        )

        if not verified:
            print(
                f"{R}"
                "✗ FINAL AUTHORIZATION FAILED"
                f"{W}"
            )
            raise SystemExit(1)

        # --------------------------------------------------
        # 7. EXECUTION
        #
        # First actual active-view motion.
        # --------------------------------------------------

        print(
            f"\n{C}"
            "▶ EXECUTING AUTHORIZED VIEWPOINT TRAJECTORY"
            f"{W}"
        )

        result = (
            robot.execute_planned_trajectory(
                selected.trajectory,
            )
        )

        print(
            f"{Y}execution success:{W}",
            result.success,
        )

        print(
            f"{Y}failure reason:{W}",
            result.failure_reason,
        )

        if not result.success:
            print(
                f"{R}"
                "✗ TRAJECTORY EXECUTION FAILED"
                f"{W}"
            )
            raise SystemExit(1)

        print(
            f"{G}"
            "✓ TRAJECTORY EXECUTED"
            f"{W}"
        )

        # --------------------------------------------------
        # 8. Verify actual wrist optical camera pose
        # --------------------------------------------------

        tf_camera = wait_for_tf(
            robot,
            "world",
            "panda_wrist_camera_optical_frame",
            timeout_s=3.0,
        )

        if tf_camera is None:
            print(
                f"{R}"
                "✗ POST-MOTION CAMERA TF UNAVAILABLE"
                f"{W}"
            )
            raise SystemExit(1)

        actual_camera_world = (
            translation(
                tf_camera
            )
        )

        camera_position_error = float(
            np.linalg.norm(
                actual_camera_world
                - desired_camera_world
            )
        )

        print(
            f"\n{C}"
            "════════ POST-MOTION VERIFICATION ════════"
            f"{W}"
        )

        print(
            f"{Y}desired camera:{W}",
            desired_camera_world
            .round(4)
            .tolist(),
        )

        print(
            f"{Y}actual camera:{W}",
            actual_camera_world
            .round(4)
            .tolist(),
        )

        print(
            f"{Y}camera position error:{W}",
            round(
                camera_position_error,
                6,
            ),
            "m",
        )

        #
        # Planning tolerance is 8 mm.
        # Allow a little additional controller/simulation
        # settling error for this first integration test.
        #
        if camera_position_error > 0.025:
            print(
                f"{R}"
                "✗ ACTIVE VIEW POSE VERIFICATION FAILED"
                f"{W}"
            )
            raise SystemExit(1)

        print(
            f"\n{G}"
            "✓ ACTIVE VIEW MOTION VERIFIED"
            f"{W}"
        )

        print(
            f"{G}"
            "✓ WRIST CAMERA REACHED SELECTED VIEWPOINT"
            f"{W}"
        )

    finally:
        robot.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
