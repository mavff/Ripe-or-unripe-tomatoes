"""Read AerialYield YOLO images using the leakage-free COCO split lists."""

import json
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import yaml
from PIL import Image

from fruit_harvest.data.types import TomatoBox, TomatoImage


STAGES = ("green", "breakers", "turning", "pink", "light red", "red")


@dataclass(frozen=True)
class AerialDataset:
    splits: dict[str, list[TomatoImage]]
    image_sources: dict[str, Path]


def _official_split_names(coco_zip: Path) -> dict[str, set[str]]:
    """The YOLO archive repeats 312 images across folders; COCO lists the real split."""
    result = {}
    with ZipFile(coco_zip) as archive:
        for split in ("train", "val", "test"):
            member = f"dataset/detection/coco/instances_{split}.json"
            payload = json.loads(archive.read(member))
            names = [Path(row["file_name"]).name for row in payload["images"]]
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate names in official {split} split")
            result[split] = set(names)
    if any(result[a] & result[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Official COCO split lists overlap")
    return result


def _stage_names(descriptor: dict) -> dict[int, str]:
    names = descriptor["names"]
    pairs = names.items() if isinstance(names, dict) else enumerate(names)
    indexed = {int(key): str(value).lower().strip() for key, value in pairs}
    ids = sorted(indexed)
    if ids not in (list(range(6)), list(range(1, 7))):
        raise ValueError(f"Expected six ordered AerialYield class IDs, got {ids}")
    if tuple(indexed[class_id] for class_id in ids) != STAGES:
        raise ValueError(f"Unexpected AerialYield stage names: {indexed}")
    return indexed


def _class_name(stage: str, ripe_stage: str) -> str:
    if stage == "green":
        return "unripe"
    if stage == "red" or (stage == "light red" and ripe_stage == "light-red"):
        return "ripe"
    return "semi_ripe"


def read_aerial_yolo(root: Path, coco_zip: Path, ripe_stage: str = "red") -> AerialDataset:
    """Filter duplicated YOLO folders by official COCO membership, then map classes."""
    if ripe_stage not in {"red", "light-red"}:
        raise ValueError("ripe_stage must be red or light-red")
    descriptor = yaml.safe_load((root / "tomato.yaml").read_text(encoding="utf-8"))
    stages = _stage_names(descriptor)
    official_names = _official_split_names(coco_zip)
    splits = {}
    image_sources = {}
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        images = [image_dir / name for name in sorted(official_names[split])]
        rows = []
        for image in images:
            if not image.is_file():
                raise FileNotFoundError(f"Official split image missing from YOLO archive: {image}")
            with Image.open(image) as loaded:
                width, height = loaded.size
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                raise FileNotFoundError(f"Missing source label {label}")
            boxes = []
            seen_boxes = set()
            for number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"Expected YOLO box at {label}:{number}")
                class_id = int(parts[0])
                if class_id not in stages:
                    raise ValueError(f"Unknown class ID {class_id} at {label}:{number}")
                cx, cy, box_width, box_height = map(float, parts[1:])
                box_key = (class_id, cx, cy, box_width, box_height)
                if box_key in seen_boxes:
                    continue  # Match Ultralytics' handling of exact duplicate source labels.
                seen_boxes.add(box_key)
                # Source coordinates are rounded to eight decimals; border boxes can differ by 1e-8.
                tolerance = 1e-6
                if not (0 < box_width <= 1 and 0 < box_height <= 1
                        and box_width / 2 - tolerance <= cx <= 1 - box_width / 2 + tolerance
                        and box_height / 2 - tolerance <= cy <= 1 - box_height / 2 + tolerance):
                    raise ValueError(f"Invalid normalized box at {label}:{number}")
                boxes.append(TomatoBox(
                    _class_name(stages[class_id], ripe_stage),
                    (cx - box_width / 2) * width,
                    (cy - box_height / 2) * height,
                    box_width * width,
                    box_height * height,
                ))
            rows.append(TomatoImage(image.name, width, height, tuple(boxes)))
            image_sources[image.name] = image
        splits[split] = rows
    return AerialDataset(splits, image_sources)
