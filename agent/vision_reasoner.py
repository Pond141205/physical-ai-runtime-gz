import base64
import json
import os
from pathlib import Path

from groq import Groq


class VisionReasoner:
    """
    Multimodal scene-understanding layer.

    This component may inspect real camera images and reason about
    spatial relationships, visibility and occlusion.

    It does NOT execute robot motion.
    """

    def __init__(
        self,
        model="qwen/qwen3.6-27b",
    ):
        api_key = os.environ.get("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY_NOT_SET"
            )

        self.model = model
        self.client = Groq(
            api_key=api_key
        )

    @staticmethod
    def _image_to_data_url(path):
        path = Path(path)

        if not path.exists():
            raise RuntimeError(
                f"IMAGE_NOT_FOUND:{path}"
            )

        suffix = path.suffix.lower()

        mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(suffix)

        if mime is None:
            raise RuntimeError(
                f"UNSUPPORTED_IMAGE_TYPE:{suffix}"
            )

        encoded = base64.b64encode(
            path.read_bytes()
        ).decode("ascii")

        return (
            f"data:{mime};base64,{encoded}"
        )

    def analyze(
        self,
        images,
        scene_context=None,
        task=None,
    ):
        """
        images:
            {
                "main": "/tmp/main.png",
                "side": "/tmp/side.png",
                "wrist": "/tmp/wrist.png",
            }

        scene_context:
            Structured runtime knowledge. This is supporting evidence,
            not a replacement for visual observation.
        """

        if not images:
            raise RuntimeError(
                "NO_IMAGES_PROVIDED"
            )

        system_prompt = """
You are the multimodal perception and spatial-reasoning layer
of a safety-first Physical AI runtime.

You are looking at REAL robot camera images.

Your job is to:
- describe what is visually observable
- identify the target object when possible
- reason about robot self-occlusion
- compare camera viewpoints
- reason about approximate spatial relationships
- identify whether more visual information is needed
- recommend a SEMANTIC next action

Do not assume that structured scene metadata is correct merely
because it was supplied. Treat images as independent evidence
and compare them with runtime metadata.

You may reason spatially, but you do NOT directly control hardware.

You MUST NOT output:
- joint commands
- motor commands
- torques
- actuator commands
- an executable trajectory

A future layer may allow candidate Cartesian poses, but this
version outputs spatial intent only.

Return JSON only.
""".strip()

        schema_description = {
            "target_visible": "boolean",
            "target_camera": (
                "main, side, wrist, multiple, or none"
            ),
            "robot_occlusion": "boolean",
            "scene_description": "string",
            "target_description": "string",
            "spatial_relationships": [
                "short natural-language relationships"
            ],
            "recommended_action": (
                "OBSERVE, REOBSERVE, CHANGE_VIEWPOINT, "
                "GRASP, VERIFY_GRASP, ABORT"
            ),
            "spatial_intent": "string",
            "confidence": "number from 0 to 1",
            "reason": "string",
        }

        text_context = {
            "task": task,
            "runtime_scene_context":
                scene_context or {},
            "required_output":
                schema_description,
        }

        content = [
            {
                "type": "text",
                "text": json.dumps(
                    text_context,
                    default=str,
                ),
            }
        ]

        for camera_name, image_path in images.items():
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"CAMERA VIEW: {camera_name}"
                    ),
                }
            )

            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url":
                            self._image_to_data_url(
                                image_path
                            )
                    },
                }
            )

        response = (
            self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content":
                            system_prompt,
                    },
                    {
                        "role": "user",
                        "content":
                            content,
                    },
                ],
                temperature=0.6,
                max_completion_tokens=1024,
                reasoning_effort="none",
                reasoning_format="hidden",
                response_format={
                    "type": "json_object"
                },
            )
        )

        raw = (
            response
            .choices[0]
            .message
            .content
        )

        if not raw:
            raise RuntimeError(
                "VISION_REASONER_EMPTY_RESPONSE"
            )

        result = json.loads(raw)

        required = {
            "target_visible",
            "target_camera",
            "robot_occlusion",
            "scene_description",
            "target_description",
            "spatial_relationships",
            "recommended_action",
            "spatial_intent",
            "confidence",
            "reason",
        }

        missing = (
            required - set(result)
        )

        if missing:
            raise RuntimeError(
                "VISION_RESPONSE_MISSING_FIELDS:"
                + ",".join(sorted(missing))
            )

        return result
