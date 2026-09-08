import unittest
from types import SimpleNamespace

from physical_ai_runtime.perception.object_tracker import ObjectTracker


def detection(label, confidence, bbox):
    return SimpleNamespace(
        object_id=f"{label}_0",
        label=label,
        confidence=confidence,
        bbox=bbox,
    )


class ObjectTrackerTest(unittest.TestCase):
    def test_same_label_keeps_track_id_and_predicts_short_gap(self):
        tracker = ObjectTracker(
            camera="main",
            iou_threshold=0.20,
            max_missed=2,
        )

        first = tracker.update(
            [detection("cube", 0.9, (10, 10, 30, 30))],
            timestamp=1.0,
        )
        predicted = tracker.update([], timestamp=1.1)
        second = tracker.update(
            [detection("cube", 0.92, (12, 10, 32, 30))],
            timestamp=1.2,
        )

        self.assertEqual(first[0].track_id, predicted[0].track_id)
        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertEqual(predicted[0].state, "PREDICTED")
        self.assertEqual(second[0].state, "DETECTED")

    def test_different_label_never_reuses_track(self):
        tracker = ObjectTracker(camera="side")
        cube = tracker.update(
            [detection("cube", 0.9, (10, 10, 30, 30))],
            timestamp=1.0,
        )[0]
        cylinder = tracker.update(
            [detection("cylinder", 0.9, (10, 10, 30, 30))],
            timestamp=1.1,
        )

        self.assertNotEqual(cube.track_id, cylinder[-1].track_id)
        self.assertEqual(cube.label, "cube")
        self.assertEqual(cylinder[-1].label, "cylinder")

    def test_stale_timestamp_fails_closed(self):
        tracker = ObjectTracker(camera="wrist")
        tracker.update(
            [detection("cube", 0.9, (1, 1, 5, 5))],
            timestamp=2.0,
        )

        with self.assertRaises(ValueError):
            tracker.update([], timestamp=1.0)


if __name__ == "__main__":
    unittest.main()
