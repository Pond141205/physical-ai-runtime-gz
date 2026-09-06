import rclpy

from grasp_verify_scene_observer import (
    GraspVerifySceneObserver,
)

C = "\033[1;36m"
G = "\033[1;32m"
Y = "\033[1;33m"
R = "\033[1;31m"
W = "\033[0m"

rclpy.init()

observer = GraspVerifySceneObserver(
    query="cube",
    reference_world=None,
    reference_size=None,
    workspace=None,
)

try:
    print(
        f"\n{C}════════ SIDE DEPTH-AWARE TEST ════════{W}"
    )

    scene = observer.observe_once(
        timeout=8.0
    )

    status = observer.last_observation_status
    metrics = observer.last_observation_metrics

    color = G if status == "OBJECT_VISIBLE" else R

    print(
        f"\n{Y}STATUS:{W} "
        f"{color}{status}{W}"
    )

    print(
        f"{Y}METRICS:{W} "
        f"{metrics}"
    )

    print(
        f"{Y}SCENE:{W} "
        f"{scene}"
    )

    print(
        f"\n{C}PERCEPTION ONLY — NO MOTION{W}"
    )

finally:
    observer.destroy_node()
    rclpy.shutdown()
