import json
import os
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image


class GeminiVisionProvider:
    def __init__(
        self,
        model="gemini-3.5-flash-lite",
        fallback_model="gemini-3.1-flash-lite",
    ):
        self._load_config()

        api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError("GEMINI_API_KEY_NOT_SET")

        self.client = genai.Client(
            api_key=api_key
        )

        self.model = model
        self.fallback_model = fallback_model
        self.last_model = None

    @staticmethod
    def _load_config():
        path = Path.home() / (
            ".config/physical-ai-runtime/gemini.env"
        )

        if not path.exists():
            return

        for raw in path.read_text().splitlines():
            line = raw.strip()

            if (
                not line
                or line.startswith("#")
                or "=" not in line
            ):
                continue

            key, value = line.split("=", 1)

            os.environ.setdefault(
                key.strip(),
                value.strip(),
            )

    def _call(
        self,
        *,
        contents,
        system_instruction,
    ):
        models = [
            self.model,
            self.fallback_model,
        ]

        last_error = None

        for model in models:
            if not model:
                continue

            try:
                response = (
                    self.client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=(
                                system_instruction
                            ),
                            temperature=0.2,
                            max_output_tokens=400,
                            response_mime_type=(
                                "application/json"
                            ),
                        ),
                    )
                )

                self.last_model = model

                if not response.text:
                    raise RuntimeError(
                        "GEMINI_EMPTY_RESPONSE"
                    )

                return json.loads(
                    response.text
                )

            except Exception as exc:
                last_error = exc
                text = str(exc).lower()

                if (
                    "429" in text
                    or "resource_exhausted" in text
                    or "rate limit" in text
                    or "quota" in text
                ):
                    continue

                raise

        raise RuntimeError(
            "GEMINI_VISION_UNAVAILABLE:"
            + str(last_error)
        )

    @staticmethod
    def _normalize_candidates(data, max_candidates):
        candidates = []
        seen = set()

        try:
            limit = max(0, int(max_candidates))
        except (TypeError, ValueError):
            limit = 8

        for item in data.get("candidates", []):
            if not isinstance(item, dict):
                continue

            label = str(item.get("label", "")).strip().lower()
            if not label or label in seen:
                continue

            try:
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0

            cameras = [
                camera
                for camera in item.get("cameras", [])
                if camera in {"main", "side", "wrist"}
            ]

            candidates.append({
                "label": label,
                "description": str(item.get("description", "")).strip(),
                "cameras": cameras,
                "confidence": max(0.0, min(1.0, confidence)),
            })
            seen.add(label)

            if len(candidates) >= limit:
                break

        return candidates

    def discover_objects(self, images, max_candidates=8):
        contents = [
            "Inspect the supplied robot camera views and propose distinct "
            "physical objects that are visibly present. Do not estimate "
            "coordinates, dimensions, poses, or robot actions."
        ]

        for camera in ("main", "side"):
            path = images.get(camera)
            if not path:
                continue

            contents.append(f"CAMERA VIEW: {camera}")
            contents.append(Image.open(path))

        system_instruction = """
You are the semantic visual discovery layer of a safety-first Physical AI system.

Return only visible physical object candidates for later deterministic grounding.
Do not include the robot, gripper, table, floor, wall, camera, shadows, or artifacts.
Do not return XYZ coordinates, poses, dimensions, trajectories, or robot commands.
Merge the same object seen by multiple cameras.

Return JSON:
{
  "scene_description": "...",
  "candidates": [
    {
      "label": "red cube",
      "description": "...",
      "cameras": ["main", "side"],
      "confidence": 0.95
    }
  ]
}
""".strip()

        parsed = self._call(
            contents=contents,
            system_instruction=system_instruction,
        )

        return {
            "scene_description": str(
                parsed.get("scene_description", "")
            ).strip(),
            "candidates": self._normalize_candidates(
                parsed,
                max_candidates,
            ),
        }

    def analyze(
        self,
        images,
        scene_context=None,
        task=None,
    ):
        contents = []

        prompt = (
            "Task: "
            + str(task or "Observe the scene.")
            + "\nScene context:\n"
            + json.dumps(
                scene_context or {},
                ensure_ascii=False,
            )
            + "\n\n"
            "Analyze the supplied robot camera views. "
            "Return visual evidence only. "
            "Do not invent XYZ coordinates or robot actions."
        )

        contents.append(prompt)

        # General conversational vision:
        # main + side only.
        for camera in ("main", "side"):
            path = images.get(camera)

            if not path:
                continue

            contents.append(
                f"CAMERA VIEW: {camera}"
            )

            contents.append(
                Image.open(path)
            )

        system_instruction = """
You are the visual perception component of a
safety-first Physical AI robotics system.

You may describe visible objects and relationships.

You do NOT control the robot.

Never invent:
- XYZ coordinates
- joint values
- trajectories
- velocities
- torque
- motion authorization

Return ONLY JSON:

{
  "target": null,
  "target_visible": false,
  "best_camera": "main",
  "cameras": {
    "main": {
      "target_visible": false,
      "robot_occlusion": false,
      "usable": true,
      "view_quality": "good",
      "confidence": 0.0,
      "reason": ""
    },
    "side": {
      "target_visible": false,
      "robot_occlusion": false,
      "usable": true,
      "view_quality": "good",
      "confidence": 0.0,
      "reason": ""
    },
    "wrist": {
      "target_visible": false,
      "robot_occlusion": false,
      "usable": false,
      "view_quality": "not_supplied",
      "confidence": 0.0,
      "reason": ""
    }
  },
  "scene_description": "",
  "target_description": "",
  "spatial_relationships": [],
  "spatial_intent": "",
  "confidence": 0.0,
  "reason": ""
}
""".strip()

        return self._call(
            contents=contents,
            system_instruction=system_instruction,
        )
