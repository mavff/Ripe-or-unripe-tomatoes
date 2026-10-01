"""Validate source images, make splits, and write one YOLO dataset layout."""

import json
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

from fruit_harvest.data.types import TomatoImage
from fruit_harvest.taxonomy import CLASS_NAMES, CLASS_TO_ID


def _image_index(root: Path) -> dict[str, Path]:
    images = {}
    for path in root.rglob("*"):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if path.name in images:
            raise ValueError(f"Duplicate image filename {path.name} under {root}")
        images[path.name] = path
    return images


def _choose_splits(train_images, test_images, validation_images, validation_fraction, seed):
    if validation_images is not None:
        return {"train": train_images, "val": validation_images, "test": test_images}
    shuffled = sorted(train_images, key=lambda image: image.file_name)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    return {"train": shuffled[validation_count:], "val": shuffled[:validation_count], "test": test_images}


def _label_lines(image: TomatoImage) -> list[str]:
    lines = []
    for box in image.boxes:
        cx = (box.x + box.width / 2) / image.width
        cy = (box.y + box.height / 2) / image.height
        width = box.width / image.width
        height = box.height / image.height
        if not all(0 <= value <= 1 for value in (cx, cy, width, height)):
            raise ValueError(f"Box outside image {image.file_name}")
        lines.append(f"{CLASS_TO_ID[box.class_name]} {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}")
    return lines


def prepare_dataset(
    images_root: Path,
    train_images: list[TomatoImage],
    test_images: list[TomatoImage],
    output: Path,
    validation_fraction: float,
    seed: int,
    source: str,
    validation_images: list[TomatoImage] | None = None,
    image_sources: dict[str, Path] | None = None,
) -> dict:
    """Preserve source splits and write images, labels, and a split manifest.

    ``image_sources`` resolves an image that appears in multiple source folders.
    """
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    splits = _choose_splits(train_images, test_images, validation_images, validation_fraction, seed)
    split_names = {split: [Path(image.file_name).name for image in rows] for split, rows in splits.items()}
    if any(len(names) != len(set(names)) for names in split_names.values()):
        raise ValueError("Duplicate image filenames in annotations")
    if (set(split_names["train"]) & set(split_names["val"])
            or set(split_names["train"]) & set(split_names["test"])
            or set(split_names["val"]) & set(split_names["test"])):
        raise ValueError("Dataset splits overlap")
    if len(splits["train"]) < 1 or not splits["val"] or not splits["test"]:
        raise ValueError("Need nonempty train, validation and test splits")

    source_images = image_sources if image_sources is not None else _image_index(images_root)
    missing = set().union(*(set(names) for names in split_names.values())) - source_images.keys()
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} source images, including {sorted(missing)[:3]}")

    counts = {}
    for split, rows in splits.items():
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        class_counts: Counter[str] = Counter()
        for image in rows:
            source_image = source_images[Path(image.file_name).name]
            with Image.open(source_image) as loaded:
                if loaded.size != (image.width, image.height):
                    raise ValueError(f"Image size disagrees with annotation: {source_image}")
            shutil.copy2(source_image, image_dir / source_image.name)
            lines = _label_lines(image)
            class_counts.update(box.class_name for box in image.boxes)
            (label_dir / f"{source_image.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
        counts[split] = {"images": len(rows), "objects": dict(class_counts)}

    dataset_yaml = {
        "path": str(output.resolve()), "train": "images/train", "val": "images/val",
        "test": "images/test", "names": dict(enumerate(CLASS_NAMES)),
    }
    (output / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")
    manifest = {"schema_version": 1, "source": source, "seed": seed, "splits": split_names, "counts": counts}
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
