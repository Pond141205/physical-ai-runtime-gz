from dataclasses import dataclass, asdict
from math import sqrt
from typing import Any, Dict, List, Optional, Tuple


VISIBLE_STATUS = "OBJECT_VISIBLE"


@dataclass
class SpatialObservation:
    camera: str
    world_xyz: Tuple[float, float, float]
    timestamp: Optional[float]
    position_sigma_m: float
    uncertainty_source: str
    geometry_status: str
    single_view_validated: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class SpatialConsistencyResult:
    camera: str
    consistent: bool
    cluster_id: Optional[int]
    support_count: int
    reason: str
    position_sigma_m: float

    def to_dict(self):
        return asdict(self)


class SpatialConsensus:
    """
    Real-world-oriented cross-camera geometric consensus.

    Important design rules:

    - No fixed absolute "5 cm" rejection rule.
    - Pairwise compatibility includes measurement uncertainty.
    - Observations too far apart in time are not compared.
    - Multi-view consensus is preferred.
    - A single camera can be accepted only if explicitly marked
      as single_view_validated by deterministic perception.
    - Fails closed when multiple observations strongly disagree.
    """

    def __init__(
        self,
        base_tolerance_m: float = 0.015,
        uncertainty_scale: float = 3.0,
        default_sigma_m: float = 0.030,
        min_sigma_m: float = 0.003,
        max_sigma_m: float = 0.150,
        max_time_delta_s: float = 2.0,
        min_multiview_support: int = 2,
    ):
        self.base_tolerance_m = float(base_tolerance_m)
        self.uncertainty_scale = float(uncertainty_scale)
        self.default_sigma_m = float(default_sigma_m)
        self.min_sigma_m = float(min_sigma_m)
        self.max_sigma_m = float(max_sigma_m)
        self.max_time_delta_s = float(max_time_delta_s)
        self.min_multiview_support = int(min_multiview_support)

    @staticmethod
    def _distance(a, b) -> float:
        return sqrt(
            (a[0] - b[0]) ** 2
            + (a[1] - b[1]) ** 2
            + (a[2] - b[2]) ** 2
        )

    def _sigma(self, value) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = self.default_sigma_m

        return max(
            self.min_sigma_m,
            min(self.max_sigma_m, value),
        )

    def _time_compatible(
        self,
        a: SpatialObservation,
        b: SpatialObservation,
    ) -> bool:
        # Missing timestamps are allowed for now,
        # but should be replaced with sensor timestamps on real hardware.
        if a.timestamp is None or b.timestamp is None:
            return True

        return (
            abs(a.timestamp - b.timestamp)
            <= self.max_time_delta_s
        )

    def _allowed_distance(
        self,
        a: SpatialObservation,
        b: SpatialObservation,
    ) -> float:
        combined_sigma = sqrt(
            a.position_sigma_m ** 2
            + b.position_sigma_m ** 2
        )

        return (
            self.base_tolerance_m
            + self.uncertainty_scale * combined_sigma
        )

    def pairwise_compatible(
        self,
        a: SpatialObservation,
        b: SpatialObservation,
    ):
        if not self._time_compatible(a, b):
            return False, {
                "reason": "TIME_INCOMPATIBLE",
                "distance_m": None,
                "allowed_distance_m": None,
            }

        distance = self._distance(
            a.world_xyz,
            b.world_xyz,
        )

        allowed = self._allowed_distance(
            a,
            b,
        )

        return distance <= allowed, {
            "reason": (
                "SPATIALLY_COMPATIBLE"
                if distance <= allowed
                else "SPATIAL_OUTLIER"
            ),
            "distance_m": distance,
            "allowed_distance_m": allowed,
        }

    def _make_observations(
        self,
        geometry: Dict[str, Dict[str, Any]],
    ) -> List[SpatialObservation]:
        observations = []

        for camera, item in geometry.items():
            if item.get("status") != VISIBLE_STATUS:
                continue

            xyz = item.get("world_xyz")

            if (
                not isinstance(xyz, (list, tuple))
                or len(xyz) != 3
            ):
                continue

            try:
                xyz = tuple(float(v) for v in xyz)
            except (TypeError, ValueError):
                continue

            raw_sigma = item.get(
                "position_sigma_m"
            )

            if raw_sigma is None:
                sigma = self.default_sigma_m
                uncertainty_source = "fallback"
            else:
                sigma = self._sigma(raw_sigma)
                uncertainty_source = item.get(
                    "uncertainty_source",
                    "observer",
                )

            observations.append(
                SpatialObservation(
                    camera=camera,
                    world_xyz=xyz,
                    timestamp=item.get("timestamp"),
                    position_sigma_m=sigma,
                    uncertainty_source=uncertainty_source,
                    geometry_status=item.get(
                        "status",
                        "UNKNOWN",
                    ),
                    single_view_validated=bool(
                        item.get(
                            "single_view_validated",
                            False,
                        )
                    ),
                )
            )

        return observations

    def evaluate(
        self,
        geometry: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:

        observations = self._make_observations(
            geometry
        )

        camera_results = {
            name: SpatialConsistencyResult(
                camera=name,
                consistent=False,
                cluster_id=None,
                support_count=0,
                reason="NO_VALID_SPATIAL_OBSERVATION",
                position_sigma_m=self.default_sigma_m,
            )
            for name in geometry
        }

        if not observations:
            return {
                "mode": "NONE",
                "safe_spatial_consensus": False,
                "consensus_cameras": [],
                "camera_results": {
                    k: v.to_dict()
                    for k, v in camera_results.items()
                },
                "pairwise": {},
                "reason": "NO_GEOMETRIC_OBSERVATIONS",
            }

        if len(observations) == 1:
            obs = observations[0]

            if obs.single_view_validated:
                camera_results[
                    obs.camera
                ] = SpatialConsistencyResult(
                    camera=obs.camera,
                    consistent=True,
                    cluster_id=0,
                    support_count=1,
                    reason="SINGLE_VIEW_VALIDATED",
                    position_sigma_m=obs.position_sigma_m,
                )

                return {
                    "mode": "SINGLE_VIEW",
                    "safe_spatial_consensus": True,
                    "consensus_cameras": [
                        obs.camera
                    ],
                    "camera_results": {
                        k: v.to_dict()
                        for k, v in camera_results.items()
                    },
                    "pairwise": {},
                    "reason": "SINGLE_VIEW_VALIDATED",
                }

            camera_results[
                obs.camera
            ] = SpatialConsistencyResult(
                camera=obs.camera,
                consistent=False,
                cluster_id=None,
                support_count=1,
                reason=(
                    "SINGLE_VIEW_REQUIRES_"
                    "STRICT_VALIDATION"
                ),
                position_sigma_m=obs.position_sigma_m,
            )

            return {
                "mode": "SINGLE_VIEW",
                "safe_spatial_consensus": False,
                "consensus_cameras": [],
                "camera_results": {
                    k: v.to_dict()
                    for k, v in camera_results.items()
                },
                "pairwise": {},
                "reason": (
                    "SINGLE_VIEW_NOT_STRICTLY_VALIDATED"
                ),
            }

        # --------------------------------------------------
        # Build pairwise compatibility graph.
        # --------------------------------------------------

        adjacency = {
            obs.camera: set()
            for obs in observations
        }

        pairwise = {}

        for i in range(len(observations)):
            for j in range(i + 1, len(observations)):
                a = observations[i]
                b = observations[j]

                compatible, details = (
                    self.pairwise_compatible(a, b)
                )

                key = f"{a.camera}<->{b.camera}"

                pairwise[key] = {
                    "compatible": compatible,
                    **details,
                }

                if compatible:
                    adjacency[a.camera].add(
                        b.camera
                    )
                    adjacency[b.camera].add(
                        a.camera
                    )

        # --------------------------------------------------
        # Connected components = candidate consensus groups.
        # --------------------------------------------------

        visited = set()
        clusters = []

        for obs in observations:
            root = obs.camera

            if root in visited:
                continue

            stack = [root]
            component = []

            while stack:
                current = stack.pop()

                if current in visited:
                    continue

                visited.add(current)
                component.append(current)

                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        stack.append(neighbor)

            clusters.append(component)

        clusters.sort(
            key=lambda c: len(c),
            reverse=True,
        )

        best_cluster = (
            clusters[0]
            if clusters
            else []
        )

        # Ambiguous equal-size independent clusters:
        # fail closed rather than arbitrarily picking one.
        if len(clusters) > 1:
            if (
                len(clusters[0])
                == len(clusters[1])
                and len(clusters[0])
                >= self.min_multiview_support
            ):
                return {
                    "mode": "MULTI_VIEW",
                    "safe_spatial_consensus": False,
                    "consensus_cameras": [],
                    "camera_results": {
                        k: v.to_dict()
                        for k, v in camera_results.items()
                    },
                    "pairwise": pairwise,
                    "reason": (
                        "AMBIGUOUS_MULTIVIEW_CLUSTERS"
                    ),
                }

        if (
            len(best_cluster)
            < self.min_multiview_support
        ):
            for obs in observations:
                camera_results[
                    obs.camera
                ] = SpatialConsistencyResult(
                    camera=obs.camera,
                    consistent=False,
                    cluster_id=None,
                    support_count=1,
                    reason="NO_MULTIVIEW_CONSENSUS",
                    position_sigma_m=obs.position_sigma_m,
                )

            return {
                "mode": "MULTI_VIEW",
                "safe_spatial_consensus": False,
                "consensus_cameras": [],
                "camera_results": {
                    k: v.to_dict()
                    for k, v in camera_results.items()
                },
                "pairwise": pairwise,
                "reason": "NO_MULTIVIEW_CONSENSUS",
            }

        obs_map = {
            o.camera: o
            for o in observations
        }

        for camera in best_cluster:
            obs = obs_map[camera]

            camera_results[
                camera
            ] = SpatialConsistencyResult(
                camera=camera,
                consistent=True,
                cluster_id=0,
                support_count=len(best_cluster),
                reason="MULTIVIEW_CONSENSUS",
                position_sigma_m=obs.position_sigma_m,
            )

        for obs in observations:
            if obs.camera in best_cluster:
                continue

            camera_results[
                obs.camera
            ] = SpatialConsistencyResult(
                camera=obs.camera,
                consistent=False,
                cluster_id=None,
                support_count=1,
                reason="SPATIAL_OUTLIER",
                position_sigma_m=obs.position_sigma_m,
            )

        return {
            "mode": "MULTI_VIEW",
            "safe_spatial_consensus": True,
            "consensus_cameras": best_cluster,
            "camera_results": {
                k: v.to_dict()
                for k, v in camera_results.items()
            },
            "pairwise": pairwise,
            "reason": "MULTIVIEW_CONSENSUS_ACCEPTED",
        }
