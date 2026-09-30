"""Stable result types shared with a future robot controller."""

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Detection:
    bbox_xyxy: tuple[float, float, float, float]
    class_name: str
    confidence: float
    decision: str

    def as_dict(self) -> dict:
        data = asdict(self)
        data["bbox_xyxy"] = [round(value, 2) for value in self.bbox_xyxy]
        data["confidence"] = round(self.confidence, 6)
        return data


@dataclass(frozen=True)
class ImageResult:
    image: Path
    width: int
    height: int
    detections: tuple[Detection, ...]

    def as_dict(self) -> dict:
        return {
            "schema_version": 1,
            "image": str(self.image),
            "width": self.width,
            "height": self.height,
            "detections": [detection.as_dict() for detection in self.detections],
        }
