import base64
import json
import mimetypes
from pathlib import Path

from groq import Groq


class MultiviewVisionReasoner:
    CAMERA_NAMES = ("main", "side", "wrist")

    def __init__(
        self,
        model="qwen/qwen3.6-27b",
        client=None,
    ):
        self.client = client or Groq()
        self.model = model

    @staticmethod
    def _image_data_url(path):
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"IMAGE_NOT_FOUND:{path}"
            )

        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "image/jpeg"

        encoded = base64.b64encode(
            path.read_bytes()
        ).decode("ascii")

        return f"data:{mime};base64,{encoded}"

    @staticmethod
    def _camera_default():
        return {
            "target_visible": False,
            "robot_occlusion": False,
            "usable": False,
            "view_quality": "unknown",
            "confidence": 0.0,
            "reason": "",
        }

    def _normalize(self, data):
        raw_cameras = data.get("cameras", {})
        cameras = {}

        for name in self.CAMERA_NAMES:
            raw = raw_cameras.get(name, {})

            if not isinstance(raw, dict):
                raw = {}

            item = {
                **self._camera_default(),
                **raw,
            }

            item["target_visible"] = bool(
                item.get("target_visible", False)
            )

            item["robot_occlusion"] = bool(
                item.get("robot_occlusion", False)
            )

            item["usable"] = bool(
                item.get("usable", False)
            )

            try:
                confidence = float(
                    item.get("confidence", 0.0)
                )
            except (TypeError, ValueError):
                confidence = 0.0

            item["confidence"] = max(
                0.0,
                min(1.0, confidence),
            )

            cameras[name] = item

        best_camera = data.get("best_camera")

        if best_camera not in self.CAMERA_NAMES:
            best_camera = None

        try:
            confidence = float(
                data.get("confidence", 0.0)
            )
        except (TypeError, ValueError):
            confidence = 0.0

        return {
            "target": data.get("target"),
            "target_visible": bool(
                data.get("target_visible", False)
            ),
            "best_camera": best_camera,
            "cameras": cameras,
            "scene_description": str(
                data.get("scene_description", "")
            ),
            "target_description": str(
                data.get("target_description", "")
            ),
            "spatial_relationships": data.get(
                "spatial_relationships",
                [],
            ),
            "spatial_intent": str(
                data.get("spatial_intent", "")
            ),
            "confidence": max(
                0.0,
                min(1.0, confidence),
            ),
            "reason": str(
                data.get("reason", "")
            ),
        }

    def analyze(
        self,
        images,
        scene_context=None,
        task=None,
    ):
        content = [
            {
                "type": "text",
                "text": (
                    "Analyze every camera independently first. "
                    "Then compare the views.\n\n"
                    f"Task: {task or 'Observe the scene.'}\n\n"
                    "Scene context:\n"
                    + json.dumps(
                        scene_context or {},
                        ensure_ascii=False,
                    )
                ),
            }
        ]

        for camera_name in self.CAMERA_NAMES:
            path = images.get(camera_name)

            if not path:
                continue

            content.append(
                {
                    "type": "text",
                    "text": (
                        f"CAMERA VIEW: {camera_name}. "
                        "The next image is from this camera."
                    ),
                }
            )

            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self._image_data_url(path)
                    },
                }
            )

        system_prompt = """
You are the visual perception and spatial reasoning component
of a Physical AI robotics system.

You inspect real robot camera images.

You provide visual evidence only.
You do not control the robot.

Rules:

1. Analyze main, side and wrist independently.
2. Never say the target is visible unless it can actually be
   visually identified in that specific image.
3. Distinguish these cases:
   - target visible
   - robot self-occlusion
   - target outside field of view
   - poor viewpoint
4. Partial robot visibility does not automatically mean that
   the target is occluded.
5. If a camera points at floor, wall, robot body, darkness or
   another irrelevant region, target_visible must be false.
6. Do not invent XYZ coordinates from the image.
7. Do not output joint angles, trajectories, motor commands,
   torque commands or executable robot motion.
8. Do not output action commands such as GRASP, MOVE_TCP,
   MOVE_JOINTS or RELEASE.
9. You may propose semantic spatial intent, for example:
   "obtain a viewpoint above and to the right of the target".
10. Motion validation and execution belong to RobotRuntime
    and MoveIt.

Return ONLY one valid JSON object with exactly this structure:

{
  "target": "cube",
  "target_visible": true,
  "best_camera": "main",
  "cameras": {
    "main": {
      "target_visible": true,
      "robot_occlusion": false,
      "usable": true,
      "view_quality": "good",
      "confidence": 0.9,
      "reason": "..."
    },
    "side": {
      "target_visible": true,
      "robot_occlusion": false,
      "usable": true,
      "view_quality": "partial",
      "confidence": 0.8,
      "reason": "..."
    },
    "wrist": {
      "target_visible": false,
      "robot_occlusion": false,
      "usable": false,
      "view_quality": "poor",
      "confidence": 0.8,
      "reason": "..."
    }
  },
  "scene_description": "...",
  "target_description": "...",
  "spatial_relationships": ["..."],
  "spatial_intent": "...",
  "confidence": 0.9,
  "reason": "..."
}

best_camera must be null if no camera provides reliable
visual evidence.

robot_occlusion=true only when the robot physically blocks a
meaningful part of the target/view.

A wrist camera pointing away from the target is not necessarily
robot occlusion.
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
            temperature=0.6,
            max_completion_tokens=700,
            reasoning_effort="none",
            reasoning_format="hidden",
            response_format={
                "type": "json_object"
            },
        )

        raw = response.choices[0].message.content

        if not raw:
            raise RuntimeError(
                "VISION_MODEL_EMPTY_RESPONSE"
            )

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"VISION_MODEL_INVALID_JSON:{raw}"
            ) from exc

        return self._normalize(parsed)
