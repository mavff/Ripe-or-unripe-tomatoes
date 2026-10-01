"""Format-neutral annotations used between source readers and YOLO preparation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TomatoBox:
    class_name: str
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class TomatoImage:
    file_name: str
    width: int
    height: int
    boxes: tuple[TomatoBox, ...]
