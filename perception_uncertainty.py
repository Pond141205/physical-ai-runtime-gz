from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PositionUncertainty:
    sigma_m: Optional[float]
    source: str


class PositionUncertaintyModel:
    """
    Interface for camera-specific metric-position uncertainty.

    Real hardware adapters can later replace this with:
      - RealSense depth-noise model
      - ZED confidence/depth model
      - calibration covariance
      - empirical per-camera calibration
    """

    def estimate(
        self,
        camera: str,
        depth_m: Optional[float],
    ) -> PositionUncertainty:
        raise NotImplementedError


class FallbackDepthUncertaintyModel(
    PositionUncertaintyModel
):
    """
    Temporary simulator / uncalibrated fallback.

    This is intentionally labeled FALLBACK and must not be
    presented as calibrated real-world sensor uncertainty.
    """

    def __init__(
        self,
        min_sigma_m=0.008,
        max_sigma_m=0.100,
        quadratic_scale=0.004,
    ):
        self.min_sigma_m = float(
            min_sigma_m
        )
        self.max_sigma_m = float(
            max_sigma_m
        )
        self.quadratic_scale = float(
            quadratic_scale
        )

    def estimate(
        self,
        camera: str,
        depth_m: Optional[float],
    ) -> PositionUncertainty:

        if depth_m is None:
            return PositionUncertainty(
                sigma_m=None,
                source=(
                    "fallback_depth_model:"
                    f"{camera}:no_depth"
                ),
            )

        try:
            depth = float(depth_m)
        except (TypeError, ValueError):
            return PositionUncertainty(
                sigma_m=None,
                source=(
                    "fallback_depth_model:"
                    f"{camera}:invalid_depth"
                ),
            )

        if depth <= 0.0:
            return PositionUncertainty(
                sigma_m=None,
                source=(
                    "fallback_depth_model:"
                    f"{camera}:invalid_depth"
                ),
            )

        sigma = (
            self.min_sigma_m
            + self.quadratic_scale
            * depth
            * depth
        )

        sigma = max(
            self.min_sigma_m,
            min(
                self.max_sigma_m,
                sigma,
            ),
        )

        return PositionUncertainty(
            sigma_m=float(sigma),
            source=(
                "fallback_depth_model:"
                f"{camera}"
            ),
        )
