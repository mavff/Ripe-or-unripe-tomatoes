"""Read AgRobTomato Pascal VOC XML and keep nearby video frames together."""

import random
import xml.etree.ElementTree as ET
from pathlib import Path

from fruit_harvest.data.types import TomatoBox, TomatoImage
from fruit_harvest.taxonomy import canonical_class


SPLIT_BLOCK_FRAMES = 20
HOLDOUT_FRACTION = 0.2


def _read_list(root: Path, list_file: Path) -> list[TomatoImage]:
    annotations = root / "Annotations"
    images = []
    for line in list_file.read_text(encoding="utf-8").splitlines():
        stem = line.split()[0] if line.strip() else ""
        if not stem:
            continue
        xml_root = ET.parse(annotations / f"{stem}.xml").getroot()
        size = xml_root.find("size")
        if size is None:
            raise ValueError(f"Missing image size in {stem}.xml")
        width = int(size.findtext("width", "0"))
        height = int(size.findtext("height", "0"))
        boxes = []
        for obj in xml_root.findall("object"):
            label = canonical_class(obj.findtext("name", ""))
            box = obj.find("bndbox")
            if box is None:
                raise ValueError(f"Missing box in {stem}.xml")
            x1 = float(box.findtext("xmin", "0"))
            y1 = float(box.findtext("ymin", "0"))
            x2 = float(box.findtext("xmax", "0"))
            y2 = float(box.findtext("ymax", "0"))
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Invalid box in {stem}.xml")
            boxes.append(TomatoBox(label, x1, y1, x2 - x1, y2 - y1))
        images.append(
            TomatoImage(xml_root.findtext("filename", f"{stem}.jpg"), width, height, tuple(boxes))
        )
    return sorted(images, key=lambda image: image.file_name)


def _grouped_split(images: list[TomatoImage], seed: int):
    """Assign 20-frame video blocks to one split to reduce near-duplicate leakage."""
    groups: dict[str, list[TomatoImage]] = {}
    for image in images:
        prefix, sequence = Path(image.file_name).stem.rsplit("_", 1)
        group = f"{prefix}_{int(sequence) // SPLIT_BLOCK_FRAMES}"
        groups.setdefault(group, []).append(image)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    group_count = max(1, round(len(keys) * HOLDOUT_FRACTION))
    test_groups = set(keys[:group_count])
    validation_groups = set(keys[group_count : 2 * group_count])
    splits = (
        [image for key in keys if key not in test_groups | validation_groups for image in groups[key]],
        [image for key in keys if key in validation_groups for image in groups[key]],
        [image for key in keys if key in test_groups for image in groups[key]],
    )
    if not all(any(box.class_name == "ripe" for image in rows for box in image.boxes) for rows in splits):
        raise ValueError("Grouped split has no ripe tomatoes in one partition; choose another seed")
    return splits


def read_voc(root: Path, seed: int = 42) -> tuple[list[TomatoImage], list[TomatoImage], list[TomatoImage]]:
    """Use official split lists when present; otherwise group the archive's video frames."""
    split_dir = root / "ImageSets" / "Main"
    train_file = split_dir / "train.txt"
    test_file = split_dir / "test.txt"
    if train_file.exists() and test_file.exists():
        return _read_list(root, train_file), [], _read_list(root, test_file)
    default_file = split_dir / "default.txt"
    if not default_file.exists():
        raise FileNotFoundError("Expected train/test.txt or default.txt in ImageSets/Main")
    return _grouped_split(_read_list(root, default_file), seed)
