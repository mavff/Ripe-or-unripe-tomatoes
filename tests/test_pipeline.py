"""Focused checks for labels, split integrity and harvest decisions."""

import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from fruit_harvest.contract import Detection
from fruit_harvest.data import canonical_class, prepare_dataset, read_aerial_yolo, read_coco
from fruit_harvest.decision import HarvestPolicy
from fruit_harvest.metrics import harvest_counts, match_detections
from fruit_harvest.model import detector_class_names, validate_detector


class DataPreparationTests(unittest.TestCase):
    def test_coco_conversion_preserves_test_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            for name in ("a.jpg", "b.jpg", "c.jpg"):
                Image.new("RGB", (100, 80), "red").save(source / name)

            def write_coco(path, names):
                payload = {
                    "categories": [{"id": 7, "name": "fully-ripe"}],
                    "images": [
                        {"id": index, "file_name": name, "width": 100, "height": 80}
                        for index, name in enumerate(names, start=1)
                    ],
                    "annotations": [
                        {"image_id": index, "category_id": 7, "bbox": [10, 20, 30, 40]}
                        for index in range(1, len(names) + 1)
                    ],
                }
                path.write_text(json.dumps(payload), encoding="utf-8")

            train = root / "train.json"
            test = root / "test.json"
            write_coco(train, ["a.jpg", "b.jpg"])
            write_coco(test, ["c.jpg"])
            output = root / "processed"
            manifest = prepare_dataset(
                source, read_coco(train), read_coco(test), output, 0.5, 42, "fixture"
            )
            self.assertEqual(manifest["splits"]["test"], ["c.jpg"])
            self.assertEqual((output / "labels/test/c.txt").read_text().strip(), "2 0.25000000 0.50000000 0.30000000 0.50000000")
            self.assertFalse(set(manifest["splits"]["train"]) & set(manifest["splits"]["val"]))

    def test_source_labels_map_to_project_classes(self):
        self.assertEqual(canonical_class("unriped"), "unripe")
        self.assertEqual(canonical_class("Breaking"), "semi_ripe")
        self.assertEqual(canonical_class("reddish"), "semi_ripe")
        self.assertEqual(canonical_class("riped"), "ripe")

    def test_aerial_uses_official_split_and_deduplicates_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            yolo = root / "yolo"
            yolo.mkdir()
            (yolo / "tomato.yaml").write_text(
                "names: [Green, Breakers, Turning, Pink, Light Red, Red]\n", encoding="utf-8"
            )
            for split, name, class_id in (("train", "a.jpg", 5), ("val", "b.jpg", 4),
                                          ("test", "c.jpg", 0), ("test", "a.jpg", 5)):
                image_dir = yolo / "images" / split
                label_dir = yolo / "labels" / split
                image_dir.mkdir(parents=True, exist_ok=True)
                label_dir.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (100, 100), "red").save(image_dir / name)
                line = f"{class_id} 0.5 0.5 0.2 0.2\n"
                (label_dir / f"{Path(name).stem}.txt").write_text(
                    line * (2 if split == "train" else 1), encoding="utf-8"
                )
            coco_zip = root / "coco.zip"
            with ZipFile(coco_zip, "w") as archive:
                for split, name in (("train", "a.jpg"), ("val", "b.jpg"), ("test", "c.jpg")):
                    archive.writestr(f"dataset/detection/coco/instances_{split}.json",
                                     json.dumps({"images": [{"file_name": name}]}))

            dataset = read_aerial_yolo(yolo, coco_zip, "red")
            self.assertEqual([image.file_name for image in dataset.splits["test"]], ["c.jpg"])
            self.assertEqual(len(dataset.splits["train"][0].boxes), 1)
            self.assertEqual(dataset.splits["train"][0].boxes[0].class_name, "ripe")
            self.assertEqual(dataset.splits["val"][0].boxes[0].class_name, "semi_ripe")
            manifest = prepare_dataset(
                yolo / "images", dataset.splits["train"], dataset.splits["test"],
                root / "processed", 0.2, 42, "fixture", dataset.splits["val"],
                dataset.image_sources,
            )
            self.assertEqual(manifest["counts"]["train"]["objects"]["ripe"], 1)


class DecisionTests(unittest.TestCase):
    def test_only_confident_ripe_is_harvested(self):
        policy = HarvestPolicy(0.7)
        self.assertEqual(policy.decide("ripe", 0.7), "harvest")
        self.assertEqual(policy.decide("ripe", 0.69), "wait")
        self.assertEqual(policy.decide("semi_ripe", 0.99), "wait")

    def test_false_harvest_and_missed_ripe_are_counted(self):
        truths = [("ripe", (0, 0, 10, 10)), ("unripe", (20, 20, 30, 30))]
        predictions = [
            Detection((0, 0, 10, 10), "unripe", 0.9, "wait"),
            Detection((20, 20, 30, 30), "ripe", 0.8, "harvest"),
        ]
        record = match_detections(truths, tuple(predictions), 0.5)
        counts = harvest_counts([record], 0.7)
        self.assertEqual((counts["tp"], counts["fp"], counts["fn"]), (0, 1, 1))


class ModelAdapterTests(unittest.TestCase):
    def test_validation_adapter_exposes_project_metrics(self):
        class Detector:
            names = {0: "unripe", 1: "semi_ripe", 2: "ripe"}

            def val(self, **kwargs):
                self.options = kwargs
                return self

            results_dict = {"metrics/mAP50(B)": 0.4}

            def summary(self, decimals):
                return [{"Class": "ripe", "Images": 2, "Instances": 3,
                         "Box-P": 0.5, "Box-R": 0.25, "Box-F1": 1 / 3,
                         "mAP50": 0.2, "mAP50-95": 0.1}]

        detector = Detector()
        aggregate, classes = validate_detector(
            detector, Path("dataset.yaml"), "val", 512, Path("evaluation")
        )
        self.assertEqual(detector_class_names(detector), ["unripe", "semi_ripe", "ripe"])
        self.assertEqual(detector.options["imgsz"], 512)
        self.assertEqual(aggregate["metrics/mAP50(B)"], 0.4)
        self.assertEqual(classes["ripe"]["instances"], 3)
        self.assertEqual(classes["ripe"]["recall"], 0.25)


if __name__ == "__main__":
    unittest.main()
