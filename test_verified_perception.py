import rclpy
from pprint import pprint

from camera_manager import CameraManager


C = "\033[1;36m"
G = "\033[1;32m"
Y = "\033[1;33m"
R = "\033[1;31m"
W = "\033[0m"


def main():
    rclpy.init()
    manager = None

    try:
        print(
            f"\n{C}"
            "════════ VERIFIED PHYSICAL-AI PERCEPTION ════════"
            f"{W}"
        )

        manager = CameraManager(query="cube")

        result = manager.observe_verified(
            query="cube",
            task=(
                "Observe the cube from main, side and wrist cameras. "
                "Evaluate visibility and viewpoint quality only. "
                "Do not propose executable motion."
            ),
            timeout=5.0,
        )

        snapshot = result["snapshot"]

        print(f"\n{C}1 — SENSOR SNAPSHOT{W}")
        print(
            f"valid={snapshot.valid} "
            f"skew={snapshot.cross_camera_skew_s}"
        )

        print(f"\n{C}2 — DETERMINISTIC GEOMETRY{W}")

        for camera in ("main", "side", "wrist"):
            geo = result.get(
                "geometry_results",
                {}
            ).get(camera)

            if geo is None:
                continue

            color = (
                G
                if geo.status == "OBJECT_VISIBLE"
                else Y
            )

            print(
                f"{color}▶ {camera.upper():5s}{W} "
                f"{geo.status}"
            )
            print(f"  {geo.metrics}")

        print(f"\n{C}3 — MULTIMODAL AI{W}")

        vision = result.get("vision") or {}

        print(
            f"AI best camera: "
            f"{vision.get('best_camera')}"
        )

        for camera in ("main", "side", "wrist"):
            ai = (
                vision.get("cameras", {})
                .get(camera, {})
            )

            print(
                f"▶ {camera.upper():5s} "
                f"visible={ai.get('target_visible')} "
                f"usable={ai.get('usable')} "
                f"confidence={ai.get('confidence')}"
            )

        print(f"\n{C}4 — EVIDENCE FUSION{W}")

        fusion = result.get("fusion") or {}
        spatial = fusion.get(
            "spatial_consensus",
            {},
        )

        print(
            f"{Y}spatial:{W} "
            f"{spatial.get('reason')}"
        )

        print(
            f"{Y}consensus cameras:{W} "
            f"{spatial.get('consensus_cameras')}"
        )

        for pair, details in spatial.get(
            "pairwise",
            {},
        ).items():
            compatible = details.get(
                "compatible"
            )

            color = G if compatible else R

            print(
                f"{color}▶ {pair}{W} "
                f"{details.get('reason')} "
                f"distance={details.get('distance_m')} "
                f"allowed={details.get('allowed_distance_m')}"
            )

        for camera in ("main", "side", "wrist"):
            item = (
                fusion.get("cameras", {})
                .get(camera, {})
            )

            accepted = item.get(
                "accepted",
                False,
            )

            color = G if accepted else R

            print(
                f"{color}▶ {camera.upper():5s}{W} "
                f"{'ACCEPT' if accepted else 'REJECT'} "
                f"score={item.get('score')} "
                f"reason={item.get('reason')}"
            )

        print(f"\n{C}5 — VERIFIED RESULT{W}")

        if result["safe_visual_evidence"]:
            print(f"{G}✓ VERIFIED{W}")
            print(
                f"{G}✓ CAMERA: "
                f"{result['best_camera'].upper()}"
                f"{W}"
            )
            print(
                f"{G}✓ SceneState AUTHORIZED{W}"
            )
            pprint(result["scene"])

        else:
            print(
                f"{R}✗ NO VERIFIED SceneState{W}"
            )
            print(
                f"reason={result['reason']}"
            )

        print(
            f"\n{C}"
            "PERCEPTION ONLY — "
            "NO ROBOT MOTION EXECUTED"
            f"{W}"
        )

    finally:
        if manager is not None:
            manager.close()

        rclpy.shutdown()


if __name__ == "__main__":
    main()
