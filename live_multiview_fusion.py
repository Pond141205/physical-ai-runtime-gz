from pprint import pprint

import rclpy

from capture_multiview import capture
from agent.multiview_vision_reasoner import MultiviewVisionReasoner
from evidence_fusion import EvidenceFusion

from rgbd_scene_observer import RGBDSceneObserver
from grasp_verify_scene_observer import GraspVerifySceneObserver
from wrist_rgbd_scene_observer import WristRGBDSceneObserver
from shared_detector import SharedOpenVocabularyDetector


C = "\033[1;36m"
G = "\033[1;32m"
Y = "\033[1;33m"
R = "\033[1;31m"
M = "\033[1;35m"
W = "\033[0m"


def heading(text):
    print(
        f"\n{C}════════════════════════════════════════════{W}"
    )
    print(f"{C}  {text}{W}")
    print(
        f"{C}════════════════════════════════════════════{W}"
    )


def geometry_record(
    observer,
    scene,
    timestamp=None,
):
    status = getattr(
        observer,
        "last_observation_status",
        None,
    )

    metrics = getattr(
        observer,
        "last_observation_metrics",
        {},
    ) or {}

    if status is None:
        status = (
            "OBJECT_VISIBLE"
            if scene is not None
            else "OBJECT_NOT_DETECTED"
        )

    world_xyz = None
    depth = None

    if scene is not None:
        obj = scene.objects.get("cube")

        if obj is None and scene.objects:
            obj = next(
                iter(scene.objects.values())
            )

        if obj is not None:
            if obj.position_world is not None:
                world_xyz = [
                    float(v)
                    for v in obj.position_world
                ]

            if obj.depth is not None:
                try:
                    depth = float(obj.depth)
                except (TypeError, ValueError):
                    depth = None

    if world_xyz is None:
        raw_xyz = metrics.get(
            "world_xyz"
        )

        if (
            isinstance(raw_xyz, (list, tuple))
            and len(raw_xyz) == 3
        ):
            world_xyz = [
                float(v)
                for v in raw_xyz
            ]

    if depth is None:
        try:
            depth = float(
                metrics.get("depth")
            )
        except (TypeError, ValueError):
            depth = None

    #
    # Current fallback uncertainty model.
    #
    # This is deliberately tagged as "fallback".
    # Real cameras should supply calibrated uncertainty
    # based on their depth sensor / calibration model.
    #
    position_sigma_m = None

    if depth is not None and depth > 0.0:
        position_sigma_m = min(
            0.10,
            max(
                0.008,
                0.008
                + 0.004 * depth * depth,
            ),
        )

    #
    # Strict single-view authorization is intentionally false.
    # It should only become true after adding real deterministic
    # checks such as workspace, size prior, calibrated depth,
    # temporal tracking and calibration validity.
    #
    single_view_validated = False

    return {
        "status": status,
        "metrics": metrics,
        "scene": scene,

        "world_xyz": world_xyz,
        "depth": depth,

        "timestamp": (
            float(timestamp)
            if timestamp is not None
            else None
        ),

        "position_sigma_m": (
            position_sigma_m
        ),

        "uncertainty_source": (
            "fallback_depth_model"
            if position_sigma_m is not None
            else "fallback"
        ),

        "single_view_validated": (
            single_view_validated
        ),
    }


def print_geometry(name, record):
    status = record["status"]

    if status == "OBJECT_VISIBLE":
        color = G
    elif status == "SELF_OCCLUDED":
        color = Y
    else:
        color = R

    print(
        f"{M}▶ {name.upper():5s}{W} "
        f"{color}{status}{W}"
    )

    metrics = record.get("metrics") or {}

    if metrics:
        print(f"  {Y}metrics:{W} {metrics}")


def main():
    # -------------------------------------------------
    # STEP 1 — capture the current raw RGB observations
    # -------------------------------------------------

    heading("STEP 1 — CAPTURE RAW CAMERA VIEWS")

    images = capture(
        timeout=5.0,
        output_dir="/tmp/physical_ai_views",
    )

    for name, path in images.items():
        print(
            f"{G}✓ {name:5s}{W} {path}"
        )

    # -------------------------------------------------
    # STEP 2 — multimodal AI visual evidence
    # -------------------------------------------------

    heading("STEP 2 — MULTIMODAL VISION AI")

    vision_reasoner = MultiviewVisionReasoner()

    vision = vision_reasoner.analyze(
        images=images,
        scene_context={
            "target_object": "cube",
            "robot": "panda",
            "available_cameras": [
                "main",
                "side",
                "wrist",
            ],
        },
        task=(
            "Assess target visibility independently in main, "
            "side and wrist views. Identify robot occlusion "
            "and determine which camera gives the best "
            "current visual evidence. Do not command motion."
        ),
    )

    print(
        f"{Y}AI best camera:{W}",
        vision.get("best_camera"),
    )

    for camera, evidence in (
        vision.get("cameras", {}).items()
    ):
        print(
            f"{M}▶ {camera.upper():5s}{W} "
            f"visible={evidence.get('target_visible')} "
            f"usable={evidence.get('usable')} "
            f"occluded={evidence.get('robot_occlusion')} "
            f"confidence={evidence.get('confidence')}"
        )

    # -------------------------------------------------
    # STEP 3 — deterministic RGB-D geometry
    # -------------------------------------------------

    heading("STEP 3 — DETERMINISTIC GEOMETRY")

    rclpy.init()

    shared_detector = SharedOpenVocabularyDetector(
        box_threshold=0.20,
        text_threshold=0.20,
    )

    observers = {}
    scenes = {}
    observation_times = {}

    try:
        observers["main"] = RGBDSceneObserver(
            query="cube",
        )

        #
        # Side discovery mode:
        # no grasp reference priors are supplied here.
        #
        observers["side"] = GraspVerifySceneObserver(
            query="cube",
            reference_world=None,
            reference_size=None,
            workspace=None,
        )

        observers["wrist"] = WristRGBDSceneObserver(
            query="cube",
        )

        for camera in (
            "main",
            "side",
            "wrist",
        ):
            print(
                f"\n{Y}Observing {camera}...{W}"
            )

            scene = observers[
                camera
            ].observe_once(
                timeout=6.0
            )

            scenes[camera] = scene

            observation_times[camera] = (
                getattr(
                    observers[camera],
                    "observation_timestamp",
                    None,
                )
            )

    finally:
        for observer in observers.values():
            try:
                observer.destroy_node()
            except Exception:
                pass

        if rclpy.ok():
            rclpy.shutdown()

    geometry = {
        camera: geometry_record(
            observers[camera],
            scenes.get(camera),
            timestamp=observation_times.get(
                camera
            ),
        )
        for camera in observers
    }

    for camera in (
        "main",
        "side",
        "wrist",
    ):
        print_geometry(
            camera,
            geometry[camera],
        )

    # -------------------------------------------------
    # STEP 4 — deterministic evidence fusion
    # -------------------------------------------------

    heading("STEP 4 — EVIDENCE FUSION")

    fusion = EvidenceFusion(
        min_ai_confidence=0.60,
    )

    result = fusion.fuse(
        vision=vision,
        geometry=geometry,
    )

    spatial = result.get(
        "spatial_consensus",
        {},
    )

    print(
        f"\n{C}SPATIAL CONSENSUS:{W} "
        f"{spatial.get('mode')} "
        f"reason={spatial.get('reason')}"
    )

    print(
        f"{Y}consensus cameras:{W} "
        f"{spatial.get('consensus_cameras')}"
    )

    pairwise = spatial.get(
        "pairwise",
        {},
    )

    for pair, details in pairwise.items():
        compatible = details.get(
            "compatible",
            False,
        )

        color = G if compatible else R

        distance = details.get(
            "distance_m"
        )

        allowed = details.get(
            "allowed_distance_m"
        )

        print(
            f"{M}▶ {pair}{W} "
            f"{color}"
            f"{'CONSISTENT' if compatible else 'OUTLIER'}"
            f"{W} "
            f"distance={distance} "
            f"allowed={allowed}"
        )

    for camera in (
        "main",
        "side",
        "wrist",
    ):
        item = result["cameras"][camera]

        color = (
            G
            if item["accepted"]
            else R
        )

        print(
            f"{M}▶ {camera.upper():5s}{W} "
            f"{color}"
            f"{'ACCEPT' if item['accepted'] else 'REJECT'}"
            f"{W} "
            f"geometry={item['geometry_status']} "
            f"score={item['score']:.3f}"
        )

        print(
            f"  {Y}reason:{W} "
            f"{item['reason']}"
        )

    # -------------------------------------------------
    # STEP 5 — safe selected SceneState
    # -------------------------------------------------

    heading("STEP 5 — SAFE VISUAL RESULT")

    best_camera = result.get(
        "best_camera"
    )

    if (
        not result.get(
            "safe_visual_evidence",
            False,
        )
        or best_camera is None
    ):
        print(
            f"{R}✗ NO CAMERA PASSED FUSION{W}"
        )
        print(
            f"{R}✗ NO SceneState AUTHORIZED{W}"
        )
        return

    selected_scene = scenes.get(
        best_camera
    )

    if selected_scene is None:
        print(
            f"{R}✗ FUSION/SCENE INCONSISTENCY{W}"
        )
        print(
            f"{R}✗ FAIL CLOSED{W}"
        )
        return

    print(
        f"{G}✓ SAFE CAMERA: "
        f"{best_camera.upper()}{W}"
    )

    print(
        f"{G}✓ GEOMETRY VERIFIED{W}"
    )

    print(
        f"{G}✓ SceneState available{W}"
    )

    print(
        f"\n{Y}Selected SceneState:{W}"
    )

    pprint(
        selected_scene,
        sort_dicts=False,
    )

    print(
        f"\n{C}PERCEPTION ONLY — NO ROBOT MOTION EXECUTED{W}"
    )


if __name__ == "__main__":
    main()
