"""Harvest policy kept separate from model inference."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HarvestPolicy:
    ripe_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not 0 < self.ripe_threshold <= 1:
            raise ValueError("ripe_threshold must be in (0, 1]")

    def decide(self, class_name: str, confidence: float) -> str:
        return "harvest" if class_name == "ripe" and confidence >= self.ripe_threshold else "wait"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"schema_version": 1, "ripe_threshold": self.ripe_threshold}, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "HarvestPolicy":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            raise ValueError("Unsupported policy schema")
        return cls(ripe_threshold=float(data["ripe_threshold"]))
