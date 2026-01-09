"""OCR service using Tesseract."""

from dataclasses import dataclass
from typing import Optional

import pytesseract
from PIL import Image


@dataclass
class OCRResult:
    """Result from OCR extraction."""

    text: str
    confidence: float
    bounding_boxes: list[dict]


class OCRService:
    """OCR service wrapper for Tesseract."""

    def __init__(self, lang: str = "eng", config: str = ""):
        """Initialize OCR service.

        Args:
            lang: Tesseract language code (e.g., 'eng', 'fra', 'deu')
            config: Additional Tesseract configuration
        """
        self.lang = lang
        self.config = config

    async def extract_text(
        self,
        image: Image.Image,
        bbox: Optional[tuple[float, float, float, float]] = None,
    ) -> str:
        """Extract text from an image.

        Args:
            image: PIL Image to process
            bbox: Optional bounding box (x1, y1, x2, y2) as normalized 0-1 coords

        Returns:
            Extracted text
        """
        if bbox:
            # Convert normalized coords to pixel coords
            width, height = image.size
            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int(bbox[2] * width)
            y2 = int(bbox[3] * height)
            image = image.crop((x1, y1, x2, y2))

        # Run OCR
        text = pytesseract.image_to_string(
            image, lang=self.lang, config=self.config
        )

        return text.strip()

    async def extract_with_confidence(self, image: Image.Image) -> OCRResult:
        """Extract text with confidence scores and bounding boxes.

        Args:
            image: PIL Image to process

        Returns:
            OCRResult with text, confidence, and bounding boxes
        """
        # Get detailed OCR data
        data = pytesseract.image_to_data(
            image, lang=self.lang, output_type=pytesseract.Output.DICT
        )

        # Compile results
        words = []
        confidences = []
        bounding_boxes = []

        n_boxes = len(data["text"])
        for i in range(n_boxes):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])

            if text and conf > 0:  # Filter empty and low-confidence
                words.append(text)
                confidences.append(conf)
                bounding_boxes.append(
                    {
                        "text": text,
                        "x": data["left"][i],
                        "y": data["top"][i],
                        "width": data["width"][i],
                        "height": data["height"][i],
                        "confidence": conf,
                    }
                )

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        return OCRResult(
            text=" ".join(words),
            confidence=avg_confidence / 100,  # Normalize to 0-1
            bounding_boxes=bounding_boxes,
        )

    async def detect_orientation(self, image: Image.Image) -> dict:
        """Detect image orientation and script.

        Args:
            image: PIL Image to process

        Returns:
            Dict with orientation info
        """
        try:
            osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
            return {
                "orientation": osd.get("orientation", 0),
                "rotate": osd.get("rotate", 0),
                "script": osd.get("script", "Latin"),
                "confidence": osd.get("orientation_conf", 0),
            }
        except pytesseract.TesseractError:
            return {
                "orientation": 0,
                "rotate": 0,
                "script": "unknown",
                "confidence": 0,
            }
