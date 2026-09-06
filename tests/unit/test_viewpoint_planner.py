import numpy as np

from physical_ai_runtime.perception.scene_state import (
    SceneObject,
    SceneState,
)

from physical_ai_runtime.planning.viewpoint_planner import (
    ViewpointPlanner,
)


C = "\033[1;36m"
G = "\033[1;32m"
R = "\033[1;31m"
Y = "\033[1;33m"
W = "\033[0m"


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

plan = planner.generate_candidates(
    scene,
    object_id="cube",

    semantic_intent=(
        "obtain an unobstructed "
        "view of the target"
    ),
)


print(
    f"\n{C}"
    "════════ VIEWPOINT CANDIDATES ════════"
    f"{W}"
)


for candidate in plan.candidates:

    status = (
        f"{G}VALID{W}"
        if candidate.valid
        else f"{R}REJECT{W}"
    )

    xyz = (
        candidate
        .camera_position_world
        .round(4)
        .tolist()
    )

    print(
        f"\n{Y}"
        f"{candidate.candidate_id}"
        f"{W}"
    )

    print(
        f"  status: {status}"
    )

    print(
        f"  direction: "
        f"{candidate.semantic_direction}"
    )

    print(
        f"  camera_world: {xyz}"
    )

    print(
        f"  distance: "
        f"{candidate.distance_to_target:.3f} m"
    )

    print(
        f"  score: "
        f"{candidate.score:.3f}"
    )

    print(
        f"  reason: "
        f"{candidate.reason}"
    )


print(
    f"\n{C}"
    "════════ PLAN RESULT ════════"
    f"{W}"
)

print(
    "safe_to_plan_motion:",
    plan.safe_to_plan_motion,
)

print(
    "reason:",
    plan.reason,
)

if plan.selected_candidate:

    print(
        "selected:",
        plan.selected_candidate.candidate_id,
    )

    print(
        "selected_camera_world:",
        plan.selected_candidate
        .camera_position_world
        .round(4)
        .tolist(),
    )


assert plan.safe_to_plan_motion

assert (
    plan.selected_candidate
    is not None
)

assert all(
    candidate.valid
    for candidate in plan.candidates
)

print(
    f"\n{G}"
    "✓ VIEWPOINT CANDIDATE GENERATION PASSED"
    f"{W}"
)

print(
    "\nCANDIDATE GENERATION ONLY — "
    "NO MOVEIT PLAN — "
    "NO ROBOT MOTION"
)
