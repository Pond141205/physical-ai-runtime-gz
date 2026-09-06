from physical_ai_runtime.planning.viewpoint_selector import ViewpointSelector


C = "\033[1;36m"
G = "\033[1;32m"
R = "\033[1;31m"
Y = "\033[1;33m"
W = "\033[0m"


selector = ViewpointSelector(
    minimum_score=0.80
)


def camera_fusion(
    *,
    accepted=False,
    score=0.0,
    reason=None,
    geometry_status=None,
):
    return {
        "accepted": accepted,
        "score": score,
        "reason": reason,
        "geometry_status": geometry_status,
    }


def camera_ai(
    *,
    visible=False,
    usable=False,
    occluded=False,
):
    return {
        "target_visible": visible,
        "usable": usable,
        "robot_occlusion": occluded,
    }


def run_case(
    name,
    perception,
    expected_action,
    expected_requires_new_viewpoint,
):
    decision = selector.select(
        perception
    )

    print(
        f"\n{C}"
        f"════════ {name} ════════"
        f"{W}"
    )

    print(
        f"{Y}action:{W} "
        f"{decision.action}"
    )

    print(
        f"{Y}camera:{W} "
        f"{decision.selected_camera}"
    )

    print(
        f"{Y}reason:{W} "
        f"{decision.reason}"
    )

    print(
        f"{Y}requires_new_viewpoint:{W} "
        f"{decision.requires_new_viewpoint}"
    )

    assert (
        decision.action
        == expected_action
    ), (
        f"expected {expected_action}, "
        f"got {decision.action}"
    )

    assert (
        decision.requires_new_viewpoint
        == expected_requires_new_viewpoint
    )

    print(
        f"{G}✓ PASS{W}"
    )


# ============================================================
# CASE A
#
# A verified current camera already exists.
# System must NOT request motion.
# ============================================================

case_a = {
    "safe_visual_evidence": True,
    "best_camera": "side",

    "fusion": {
        "cameras": {
            "main": camera_fusion(
                accepted=True,
                score=0.95,
                reason="ACCEPTED",
                geometry_status="OBJECT_VISIBLE",
            ),
            "side": camera_fusion(
                accepted=True,
                score=0.99,
                reason="ACCEPTED",
                geometry_status="OBJECT_VISIBLE",
            ),
            "wrist": camera_fusion(
                accepted=False,
                score=0.0,
                reason="REJECT_SPATIAL_OUTLIER",
                geometry_status="OBJECT_VISIBLE",
            ),
        }
    },

    "vision": {
        "cameras": {
            "main": camera_ai(
                visible=True,
                usable=True,
            ),
            "side": camera_ai(
                visible=True,
                usable=True,
            ),
            "wrist": camera_ai(
                visible=False,
                usable=False,
            ),
        }
    },
}


# ============================================================
# CASE B
#
# AI claims it sees the target,
# but deterministic geometry cannot verify it.
#
# AI must NOT authorize perception.
# ============================================================

case_b = {
    "safe_visual_evidence": False,
    "best_camera": None,

    "fusion": {
        "cameras": {
            "main": camera_fusion(
                accepted=False,
                score=0.0,
                reason="REJECT_GEOMETRY",
                geometry_status="DEPTH_INVALID",
            ),
            "side": camera_fusion(
                accepted=False,
                score=0.0,
                reason="REJECT_GEOMETRY",
                geometry_status="NO_VALID_GEOMETRY",
            ),
            "wrist": camera_fusion(
                accepted=False,
                score=0.0,
                reason="REJECT_GEOMETRY",
                geometry_status="NO_VALID_GEOMETRY",
            ),
        }
    },

    "vision": {
        "cameras": {
            "main": camera_ai(
                visible=True,
                usable=True,
            ),
            "side": camera_ai(),
            "wrist": camera_ai(),
        },

        "spatial_intent":
            "obtain a geometrically verifiable view",
    },
}


# ============================================================
# CASE C
#
# Target/view is explicitly blocked by the robot.
#
# System may REQUEST another viewpoint,
# but still does NOT execute motion.
# ============================================================

case_c = {
    "safe_visual_evidence": False,
    "best_camera": None,

    "fusion": {
        "cameras": {
            "main": camera_fusion(
                accepted=False,
                score=0.0,
                reason="REJECT_OCCLUSION",
                geometry_status="SELF_OCCLUDED",
            ),
            "side": camera_fusion(
                accepted=False,
                score=0.0,
                reason="REJECT_OCCLUSION",
                geometry_status="OBJECT_OCCLUDED",
            ),
            "wrist": camera_fusion(
                accepted=False,
                score=0.0,
                reason="REJECT_GEOMETRY",
                geometry_status="NO_VALID_GEOMETRY",
            ),
        }
    },

    "vision": {
        "cameras": {
            "main": camera_ai(
                visible=False,
                usable=False,
                occluded=True,
            ),
            "side": camera_ai(
                visible=False,
                usable=False,
                occluded=True,
            ),
            "wrist": camera_ai(),
        },

        "spatial_intent":
            "obtain an unobstructed view of the target",
    },
}


run_case(
    "CASE A — VERIFIED CURRENT VIEW",
    case_a,
    "USE_CURRENT_VIEW",
    False,
)

run_case(
    "CASE B — AI ONLY / GEOMETRY UNVERIFIED",
    case_b,
    "PERCEPTION_UNVERIFIED",
    True,
)

run_case(
    "CASE C — OCCLUDED",
    case_c,
    "REQUEST_ALTERNATE_VIEW",
    True,
)


print(
    f"\n{G}"
    "════════ ALL VIEWPOINT SAFETY CASES PASSED ════════"
    f"{W}"
)

print(
    "DECISION LOGIC ONLY — "
    "NO ROBOT MOTION EXECUTED"
)
