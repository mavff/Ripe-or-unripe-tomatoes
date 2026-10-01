"""Read tomatOD COCO boxes without choosing train, validation, or test membership."""

import json
from pathlib import Path

from fruit_harvest.data.types import TomatoBox, TomatoImage
from fruit_harvest.taxonomy import canonical_class


def read_coco(path: Path) -> list[TomatoImage]:
    """Return source images and validated boxes in project-neutral types."""
    data = json.loads(path.read_text(encoding="utf-8"))
    categories = {entry["id"]: canonical_class(entry["name"]) for entry in data["categories"]}
    images = {}
    for entry in data["images"]:
        image_id = entry["id"]
        if image_id in images:
            raise ValueError(f"Duplicate image id {image_id} in {path}")
        images[image_id] = entry

    boxes_by_image: dict[int, list[TomatoBox]] = {image_id: [] for image_id in images}
    for entry in data["annotations"]:
        image_id = entry["image_id"]
        if image_id not in images:
            raise ValueError(f"Annotation refers to missing image {image_id}")
        if entry.get("iscrowd", 0):
            continue
        image = images[image_id]
        x, y, width, height = map(float, entry["bbox"])
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid box in image {image_id}")
        x2 = min(x + width, image["width"])
        y2 = min(y + height, image["height"])
        x, y = max(x, 0.0), max(y, 0.0)
        if x2 <= x or y2 <= y:
            raise ValueError(f"Box outside image {image_id}")
        boxes_by_image[image_id].append(
            TomatoBox(categories[entry["category_id"]], x, y, x2 - x, y2 - y)
        )

    result = [
        TomatoImage(
            file_name=entry["file_name"],
            width=int(entry["width"]),
            height=int(entry["height"]),
            boxes=tuple(boxes_by_image[image_id]),
        )
        for image_id, entry in images.items()
    ]
    return sorted(result, key=lambda image: image.file_name)
