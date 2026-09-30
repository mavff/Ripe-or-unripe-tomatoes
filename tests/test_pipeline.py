"""Focused checks for labels, split integrity and harvest decisions."""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from fruit_harvest.contract import Detection
from fruit_harvest.data import canonical_class, prepare_dataset, read_coco
from fruit_harvest.decision import HarvestPolicy
from fruit_harvest.metrics import harvest_counts, match_detections


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


if __name__ == "__main__":
    unittest.main()
