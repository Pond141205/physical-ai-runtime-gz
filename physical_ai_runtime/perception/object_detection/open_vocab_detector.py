from typing import List
import numpy as np
import torch
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)

from .base_detector import BaseObjectDetector
from .detection_types import Detection


class OpenVocabularyDetector(BaseObjectDetector):

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Detector initialization is intentionally silent.
        # User-facing progress is reported by RuntimeStatus.

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.model = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(model_id)
            .to(self.device)
        )

        self.model.eval()

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def detect(
        self,
        image_rgb: np.ndarray,
        query: str,
    ) -> List[Detection]:

        image = Image.fromarray(image_rgb)

        text = query.strip().lower()

        if not text.endswith("."):
            text += "."

        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        detections = []

        for i, (box, score, label) in enumerate(
            zip(
                result["boxes"],
                result["scores"],
                result["labels"],
            )
        ):
            x1, y1, x2, y2 = box.tolist()

            detections.append(
                Detection(
                    object_id=f"{query}_{i}",
                    label=str(label),
                    confidence=float(score),
                    bbox=(
                        int(round(x1)),
                        int(round(y1)),
                        int(round(x2)),
                        int(round(y2)),
                    ),
                )
            )

        return detections
