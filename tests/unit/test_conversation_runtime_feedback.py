import unittest

from physical_ai_runtime.ai.conversation_agent import (
    PhysicalAIConversationAgent,
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


if __name__ == "__main__":
    unittest.main()
