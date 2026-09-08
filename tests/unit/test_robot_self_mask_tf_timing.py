import unittest

from physical_ai_runtime.perception.robot_self_mask import RobotSelfMask


class _Logger:
    def debug(self, _message):
        pass


class _Node:
    def get_logger(self):
        return _Logger()


class _TransformBuffer:
    def __init__(self, accepted_nanoseconds):
        self.accepted_nanoseconds = accepted_nanoseconds
        self.requests = []
        self.transform = object()

    def lookup_transform(self, _camera, _link, time):
        self.requests.append(time.nanoseconds)

        if time.nanoseconds != self.accepted_nanoseconds:
            raise RuntimeError("transform unavailable")

        return self.transform


class RobotSelfMaskTfTimingTest(unittest.TestCase):
    def _mask(self, buffer, max_tf_lag_s=0.05):
        mask = RobotSelfMask.__new__(RobotSelfMask)
        mask.node = _Node()
        mask.tf_buffer = buffer
        mask.camera_frame = "camera"
        mask.max_tf_lag_s = max_tf_lag_s
        return mask

    def test_uses_bounded_future_transform_after_exact_timestamp_miss(self):
        requested = 12_000_000_000
        buffer = _TransformBuffer(
            accepted_nanoseconds=requested + 50_000_000,
        )

        transform = self._mask(buffer)._transform_at_sensor_time(
            "link",
            12.0,
        )

        self.assertIs(transform, buffer.transform)
        self.assertEqual(
            buffer.requests,
            [requested, requested + 50_000_000],
        )

    def test_refuses_when_no_transform_exists_within_bound(self):
        buffer = _TransformBuffer(accepted_nanoseconds=99_000_000_000)

        with self.assertRaisesRegex(
            RuntimeError,
            "SELF_MASK_TF_TIME_UNAVAILABLE",
        ):
            self._mask(buffer)._transform_at_sensor_time("link", 12.0)


if __name__ == "__main__":
    unittest.main()
