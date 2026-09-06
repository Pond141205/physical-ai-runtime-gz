import rclpy

from wrist_rgbd_scene_observer import WristRGBDSceneObserver

C = "\033[1;36m"
G = "\033[1;32m"
Y = "\033[1;33m"
R = "\033[1;31m"
W = "\033[0m"

rclpy.init()

observer = WristRGBDSceneObserver(
    query="cube",
)

try:
    print(
        f"\n{C}════════ WRIST GEOMETRY TEST ════════{W}"
    )

    scene = observer.observe_once(
        timeout=8.0
    )

    status = observer.last_observation_status
    metrics = observer.last_observation_metrics

    color = (
        G
        if status == "OBJECT_VISIBLE"
        else R
    )

    print(
        f"\n{Y}STATUS:{W} "
        f"{color}{status}{W}"
    )

    print(
        f"{Y}METRICS:{W} "
        f"{metrics}"
    )

    if scene is not None:
        obj = scene.objects.get("cube")

        if obj is not None:
            print(
                f"{Y}WORLD XYZ:{W} "
                f"{obj.position_world}"
            )

            print(
                f"{Y}ROBOT XYZ:{W} "
                f"{obj.position_robot}"
            )

            print(
                f"{Y}DEPTH:{W} "
                f"{obj.depth}"
            )

            print(
                f"{Y}SIZE:{W} "
                f"{obj.size_xyz}"
            )

    print(
        f"\n{C}PERCEPTION ONLY — NO MOTION{W}"
    )

finally:
    observer.destroy_node()
    rclpy.shutdown()
