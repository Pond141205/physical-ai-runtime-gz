import unittest

from physical_ai_runtime.ai.agent.multiview_vision_reasoner import (
    MultiviewVisionReasoner,
)
from physical_ai_runtime.ai.vision.gemini_provider import (
    GeminiVisionProvider,
)


class _GeminiDiscovery:
    def __init__(self):
        self.calls = []

    def discover_objects(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "scene_description": "A cube is visible.",
            "candidates": [
                {
                    "label": "cube",
                    "description": "small red cube",
                    "cameras": ["main"],
                    "confidence": 0.9,
                },
            ],
        }


class MultiviewVisionReasonerTest(unittest.TestCase):
    def test_discovery_uses_gemini_object_schema(self):
        reasoner = MultiviewVisionReasoner(client=object())
        provider = _GeminiDiscovery()
        reasoner.gemini = provider

        result = reasoner.discover_objects(
            {},
            max_candidates=3,
        )

        self.assertEqual(result["candidates"][0]["label"], "cube")
        self.assertEqual(provider.calls[0]["max_candidates"], 3)


class GeminiVisionProviderTest(unittest.TestCase):
    def test_normalizes_discovery_candidates_without_network(self):
        provider = GeminiVisionProvider.__new__(GeminiVisionProvider)
        provider._call = lambda **_: {
            "scene_description": "two objects",
            "candidates": [
                {
                    "label": " Cube ",
                    "description": "red cube",
                    "cameras": ["main", "invalid"],
                    "confidence": 1.4,
                },
                {
                    "label": "cube",
                    "description": "duplicate",
                    "cameras": ["side"],
                    "confidence": 0.2,
                },
            ],
        }

        result = provider.discover_objects({}, max_candidates=8)

        self.assertEqual(result["scene_description"], "two objects")
        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(result["candidates"][0]["label"], "cube")
        self.assertEqual(result["candidates"][0]["cameras"], ["main"])
        self.assertEqual(result["candidates"][0]["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
