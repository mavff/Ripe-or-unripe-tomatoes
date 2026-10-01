"""Record why AerialYield YOLO folders must be filtered by COCO memberships."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile


SPLITS = ("train", "val", "test")


def audit(yolo_zip: Path, coco_zip: Path) -> dict:
    with ZipFile(yolo_zip) as yolo, ZipFile(coco_zip) as coco:
        yolo_images = {split: {} for split in SPLITS}
        for member in yolo.namelist():
            parts = Path(member).parts
            if len(parts) >= 2 and parts[-3:-1] in (("images", "train"),
                                                     ("images", "val"), ("images", "test")):
                yolo_images[parts[-2]][parts[-1]] = member

        official = {}
        for split in SPLITS:
            path = f"dataset/detection/coco/instances_{split}.json"
            payload = json.loads(coco.read(path))
            official[split] = {Path(item["file_name"]).name for item in payload["images"]}
            missing = official[split] - yolo_images[split].keys()
            if missing:
                raise ValueError(f"{split}: {len(missing)} official images absent from YOLO folder")

        locations = defaultdict(list)
        for split in SPLITS:
            for name, member in yolo_images[split].items():
                locations[name].append(member)
        duplicates = {name: members for name, members in locations.items() if len(members) > 1}
        identical = sum(len({hashlib.md5(yolo.read(member)).digest() for member in members}) == 1
                        for members in duplicates.values())
        duplicate_labels = {}
        for split in SPLITS:
            count = 0
            for name in official[split]:
                member = f"dataset/detection/yolo/labels/{split}/{Path(name).stem}.txt"
                lines = yolo.read(member).decode("utf-8").splitlines()
                count += len(lines) - len(set(lines))
            duplicate_labels[split] = count
        overlap = {f"{a}_{b}": len(official[a] & official[b])
                   for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
        return {
            "source": "AerialYield-T2M release 1.0.0",
            "source_url": "https://zenodo.org/records/22071809",
            "yolo_folder_images": {split: len(yolo_images[split]) for split in SPLITS},
            "yolo_unique_image_names": len(locations),
            "repeated_names_between_yolo_folders": len(duplicates),
            "repeated_names_with_identical_bytes": identical,
            "official_coco_split_images": {split: len(official[split]) for split in SPLITS},
            "extra_yolo_entries_ignored": {
                split: len(yolo_images[split]) - len(official[split]) for split in SPLITS
            },
            "exact_duplicate_label_rows_removed": duplicate_labels,
            "official_coco_pairwise_overlap": overlap,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-zip", type=Path, default=Path("data/raw/aerial-yolo.zip"))
    parser.add_argument("--coco-zip", type=Path, default=Path("data/raw/aerial-coco.zip"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/aerial/metrics/source_split_audit.json"))
    args = parser.parse_args()
    result = audit(args.yolo_zip, args.coco_zip)
    if any(result["official_coco_pairwise_overlap"].values()):
        raise ValueError("Official split lists overlap")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
