import rclpy

from physical_ai_runtime.perception.camera_manager import CameraManager


C = "\033[1;36m"
G = "\033[1;32m"
Y = "\033[1;33m"
W = "\033[0m"


def main():
    rclpy.init()
    manager = None

    try:
        manager = CameraManager(
            query="cube"
        )

        result = manager.observe_verified(
            query="cube",
            timeout=5.0,
        )

        decision = result[
            "viewpoint_decision"
        ]

        print(
            f"\n{C}"
            "════════ VIEWPOINT DECISION ════════"
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

        print(
            f"{Y}semantic_intent:{W} "
            f"{decision.preferred_semantic_intent}"
        )

        if decision.action == "USE_CURRENT_VIEW":
            print(
                f"\n{G}"
                "✓ CURRENT VIEW SUFFICIENT"
                f"{W}"
            )

        print(
            f"\n{C}"
            "DECISION ONLY — "
            "NO VIEWPOINT MOTION EXECUTED"
            f"{W}"
        )

    finally:
        if manager is not None:
            manager.close()

        rclpy.shutdown()


if __name__ == "__main__":
    main()
