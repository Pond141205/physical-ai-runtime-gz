from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class TrackSnapshot:
    """Immutable state exposed to perception consumers."""

    track_id: str
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    timestamp: float
    hits: int
    missed: int
    confirmed: bool
    state: str


@dataclass(frozen=True)
class ContinuousPerceptionFrame:
    """Latest detector/tracker result from one perception worker cycle."""

    query: str
    observation_timestamp: Optional[float]
    received_monotonic: float
    cameras: Dict[str, Tuple[TrackSnapshot, ...]]
    inference_results: Dict[str, Any]
    snapshot: Any
    detector_latency_s: Optional[float]
    valid: bool
    reason: str

    def age_s(self, now_monotonic: float) -> float:
        return max(
            0.0,
            float(now_monotonic) - self.received_monotonic,
        )


@dataclass
class _Track:
    track_id: str
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    timestamp: float
    hits: int
    missed: int
    velocity_xy: Tuple[float, float]
    state: str


class ObjectTracker:
    """Small deterministic detect-then-track association layer.

    The tracker never invents labels.  A label enters a track only from a
    detector result, and association is allowed only between equal labels.
    """

    def __init__(
        self,
        camera: str,
        iou_threshold: float = 0.30,
        max_missed: int = 3,
        min_hits: int = 1,
    ):
        if not camera:
            raise ValueError("camera must be non-empty")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        if max_missed < 0:
            raise ValueError("max_missed must be non-negative")
        if min_hits < 1:
            raise ValueError("min_hits must be positive")

        self.camera = str(camera)
        self.iou_threshold = float(iou_threshold)
        self.max_missed = int(max_missed)
        self.min_hits = int(min_hits)
        self._tracks: Dict[str, _Track] = {}
        self._next_id = 1
        self._last_timestamp: Optional[float] = None

    @staticmethod
    def _center(bbox):
        x1, y1, x2, y2 = bbox
        return (
            0.5 * (x1 + x2),
            0.5 * (y1 + y2),
        )

    @staticmethod
    def _iou(left, right):
        lx1, ly1, lx2, ly2 = left
        rx1, ry1, rx2, ry2 = right

        ix1 = max(lx1, rx1)
        iy1 = max(ly1, ry1)
        ix2 = min(lx2, rx2)
        iy2 = min(ly2, ry2)

        intersection = max(0.0, ix2 - ix1) * max(
            0.0,
            iy2 - iy1,
        )
        left_area = max(0.0, lx2 - lx1) * max(
            0.0,
            ly2 - ly1,
        )
        right_area = max(0.0, rx2 - rx1) * max(
            0.0,
            ry2 - ry1,
        )
        union = left_area + right_area - intersection

        return 0.0 if union <= 0.0 else intersection / union

    @staticmethod
    def _label(detection):
        label = getattr(detection, "label", None)
        if label is None or not str(label).strip():
            label = getattr(detection, "object_id", "")
        return str(label).strip().casefold()

    @staticmethod
    def _bbox(detection):
        values = tuple(float(value) for value in detection.bbox)
        if len(values) != 4 or values[2] <= values[0] or values[3] <= values[1]:
            raise ValueError("detection bbox must have positive area")
        return values

    def reset(self):
        self._tracks.clear()
        self._last_timestamp = None

    def _new_track(self, detection, timestamp):
        label = self._label(detection)
        track_id = f"{self.camera}:{self._next_id}"
        self._next_id += 1
        self._tracks[track_id] = _Track(
            track_id=track_id,
            label=label,
            confidence=float(detection.confidence),
            bbox=self._bbox(detection),
            timestamp=float(timestamp),
            hits=1,
            missed=0,
            velocity_xy=(0.0, 0.0),
            state="DETECTED",
        )

    def _snapshot(self, track):
        bbox = tuple(int(round(value)) for value in track.bbox)
        return TrackSnapshot(
            track_id=track.track_id,
            label=track.label,
            confidence=max(0.0, min(1.0, track.confidence)),
            bbox=bbox,
            timestamp=track.timestamp,
            hits=track.hits,
            missed=track.missed,
            confirmed=track.hits >= self.min_hits,
            state=track.state,
        )

    def update(self, detections: Iterable[Any], timestamp: float):
        timestamp = float(timestamp)
        if (
            self._last_timestamp is not None
            and timestamp < self._last_timestamp
        ):
            raise ValueError("tracker timestamp moved backwards")

        detections = list(detections or [])
        detection_boxes = []
        for detection in detections:
            detection_boxes.append((self._label(detection), self._bbox(detection)))

        available_tracks = list(self._tracks.values())
        pairs = []
        for track in available_tracks:
            for detection_index, (label, bbox) in enumerate(detection_boxes):
                if track.label != label:
                    continue
                pairs.append((
                    -self._iou(track.bbox, bbox),
                    track.track_id,
                    detection_index,
                ))

        matched_tracks = set()
        matched_detections = set()
        for negative_iou, track_id, detection_index in sorted(pairs):
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            if -negative_iou < self.iou_threshold:
                continue

            track = self._tracks[track_id]
            detection = detections[detection_index]
            old_center = self._center(track.bbox)
            new_bbox = detection_boxes[detection_index][1]
            new_center = self._center(new_bbox)
            dt = max(1.0e-6, timestamp - track.timestamp)
            track.velocity_xy = (
                (new_center[0] - old_center[0]) / dt,
                (new_center[1] - old_center[1]) / dt,
            )
            track.bbox = new_bbox
            track.confidence = float(detection.confidence)
            track.timestamp = timestamp
            track.hits += 1
            track.missed = 0
            track.state = "DETECTED"
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)

        for track in list(self._tracks.values()):
            if track.track_id in matched_tracks:
                continue

            dt = max(0.0, timestamp - track.timestamp)
            dx = track.velocity_xy[0] * dt
            dy = track.velocity_xy[1] * dt
            x1, y1, x2, y2 = track.bbox
            track.bbox = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
            track.timestamp = timestamp
            track.missed += 1
            track.confidence *= 0.85
            track.state = "PREDICTED"

            if track.missed > self.max_missed:
                del self._tracks[track.track_id]

        for detection_index, detection in enumerate(detections):
            if detection_index not in matched_detections:
                self._new_track(detection, timestamp)

        self._last_timestamp = timestamp
        return tuple(
            self._snapshot(track)
            for track in sorted(
                self._tracks.values(),
                key=lambda item: item.track_id,
            )
        )

    def snapshot(self):
        return tuple(
            self._snapshot(track)
            for track in sorted(
                self._tracks.values(),
                key=lambda item: item.track_id,
            )
        )
