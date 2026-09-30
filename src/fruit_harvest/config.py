"""Validated, shared project configuration."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Settings:
    """One YAML-backed set of data, training, and evaluation settings."""

    model: str = "yolo11n.pt"
    image_size: int = 512
    batch_size: int = 4
    max_hours: float = 2.0
    patience: int = 7
    workers: int = 2
    seed: int = 42
    optimizer: str = "AdamW"
    initial_learning_rate: float = 0.001
    final_lr_fraction: float = 0.01
    weight_decay: float = 0.0005
    cosine_lr: bool = True
    class_weight_power: float = 0.25
    validation_fraction: float = 0.2
    prediction_confidence: float = 0.05
    match_iou: float = 0.5

    @classmethod
    def load(cls, path: Path) -> "Settings":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Expected a YAML mapping in {path}")
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown settings: {sorted(unknown)}")
        settings = cls(**raw)
        if settings.image_size <= 0 or settings.image_size % 32:
            raise ValueError("image_size must be a positive multiple of 32")
        if settings.batch_size < 1 or settings.max_hours <= 0:
            raise ValueError("batch_size and max_hours must be positive")
        if settings.optimizer != "AdamW":
            raise ValueError("This training profile supports optimizer=AdamW")
        if not isinstance(settings.cosine_lr, bool):
            raise ValueError("cosine_lr must be true or false")
        if not isfinite(settings.initial_learning_rate) or settings.initial_learning_rate <= 0:
            raise ValueError("initial_learning_rate must be positive and finite")
        if not isfinite(settings.final_lr_fraction) or not 0 < settings.final_lr_fraction <= 1:
            raise ValueError("final_lr_fraction must be in (0, 1]")
        if not isfinite(settings.weight_decay) or settings.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative and finite")
        if not isfinite(settings.class_weight_power) or not 0 <= settings.class_weight_power <= 1:
            raise ValueError("class_weight_power must be between 0 and 1")
        if not 0 < settings.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if not 0 < settings.prediction_confidence < 1:
            raise ValueError("prediction_confidence must be between 0 and 1")
        if not 0 < settings.match_iou <= 1:
            raise ValueError("match_iou must be between 0 and 1")
        return settings
