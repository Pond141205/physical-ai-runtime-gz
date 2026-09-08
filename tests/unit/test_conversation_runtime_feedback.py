import json
import unittest
from types import SimpleNamespace

from physical_ai_runtime.ai.conversation_agent import (
    PhysicalAIConversationAgent,
)


class _FakeClient:
    def __init__(self, value):
        self.messages = None
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create),
        )
        self.value = value

    def _create(self, **kwargs):
        self.messages = kwargs["messages"]
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(self.value),
                ),
            )],
        )


class ConversationRuntimeFeedbackTest(unittest.TestCase):
    def test_feedback_is_authoritative_system_context(self):
        agent = PhysicalAIConversationAgent(client=object())

        agent.record_runtime_feedback({
            "result_success": False,
            "failure_reason": "UNREACHABLE",
            "tcp_position": [0.1, 0.2, 0.3],
        })

        message = agent.history[-1]
        self.assertEqual(message["role"], "system")
        self.assertIn("AUTHORITATIVE RUNTIME FEEDBACK", message["content"])
        self.assertIn("UNREACHABLE", message["content"])
        self.assertIn("do not claim success", message["content"])


    def test_object_request_can_ask_for_observation_without_motion(self):
        client = _FakeClient({
            "mode": "observe",
            "message": "I need verified cube geometry.",
            "query": "cube",
        })
        agent = PhysicalAIConversationAgent(client=client)

        turn = agent.respond("pick up the cube")

        self.assertEqual(turn.mode, "observe")
        self.assertEqual(turn.query, "cube")
        self.assertIsNone(turn.program)

    def test_observe_requires_a_nonempty_query(self):
        client = _FakeClient({
            "mode": "observe",
            "message": "Looking",
            "query": " ",
        })
        agent = PhysicalAIConversationAgent(client=client)

        with self.assertRaisesRegex(
            RuntimeError,
            "CONVERSATION_OBSERVE_QUERY_REQUIRED",
        ):
            agent.respond("find it")


if __name__ == "__main__":
    unittest.main()
