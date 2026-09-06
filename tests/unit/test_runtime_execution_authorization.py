"""Pure regression for the runtime execution-authorization boundary."""

import unittest
from unittest.mock import patch

from physical_ai_runtime.core.runtime import RobotRuntime
from physical_ai_runtime.planning.execution_authorization import (
    ExecutionAuthorization,
)


class FakeRobot:
    def __init__(self):
        self.executed = []

    def execute_planned_trajectory(self, trajectory):
        self.executed.append(trajectory)
        return "EXECUTED"


class FakeGate:
    authorize_result = None
    verify_result = (True, "AUTHORIZED_TRAJECTORY_VERIFIED")

    def __init__(self, robot):
        self.robot = robot

    def authorize(self, trajectory, *, planned_at_monotonic):
        self.trajectory = trajectory
        self.planned_at_monotonic = planned_at_monotonic
        return self.authorize_result

    def verify_authorization(self, trajectory, authorization):
        assert trajectory is self.trajectory
        assert authorization is self.authorize_result
        return self.verify_result


def authorization(authorized, reason):
    return ExecutionAuthorization(
        authorized=authorized,
        reason=reason,
        trajectory_digest=None,
        start_state_error=0.0,
        trajectory_age_s=0.0,
        safety_context={},
        authorized_at_monotonic=0.0,
    )


class RuntimeExecutionAuthorizationTest(unittest.TestCase):
    def setUp(self):
        self.robot = FakeRobot()
        self.trajectory = object()

    def run_runtime_execution(self):
        with patch(
            "physical_ai_runtime.core.runtime.ExecutionAuthorizationGate",
            FakeGate,
        ):
            return RobotRuntime._execute_authorized_trajectory(
                self.robot,
                self.trajectory,
                123.0,
            )

    def test_executes_only_after_authorization_and_binding(self):
        FakeGate.authorize_result = authorization(
            True,
            "EXECUTION_AUTHORIZED",
        )
        FakeGate.verify_result = (
            True,
            "AUTHORIZED_TRAJECTORY_VERIFIED",
        )

        result, error = self.run_runtime_execution()

        self.assertEqual(result, "EXECUTED")
        self.assertIsNone(error)
        self.assertEqual(self.robot.executed, [self.trajectory])

    def test_denied_authorization_never_executes(self):
        FakeGate.authorize_result = authorization(
            False,
            "SAFETY_CONTEXT_FAILED:SUPPORT_SURFACE_MISSING",
        )

        result, error = self.run_runtime_execution()

        self.assertIsNone(result)
        self.assertEqual(
            error,
            "EXECUTION_AUTHORIZATION_DENIED:"
            "SAFETY_CONTEXT_FAILED:SUPPORT_SURFACE_MISSING",
        )
        self.assertEqual(self.robot.executed, [])

    def test_changed_or_stale_authorization_never_executes(self):
        FakeGate.authorize_result = authorization(
            True,
            "EXECUTION_AUTHORIZED",
        )
        FakeGate.verify_result = (
            False,
            "TRAJECTORY_CHANGED_AFTER_AUTHORIZATION",
        )

        result, error = self.run_runtime_execution()

        self.assertIsNone(result)
        self.assertEqual(
            error,
            "EXECUTION_AUTHORIZATION_INVALID:"
            "TRAJECTORY_CHANGED_AFTER_AUTHORIZATION",
        )
        self.assertEqual(self.robot.executed, [])


if __name__ == "__main__":
    unittest.main()
