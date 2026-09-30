"""Spatial matching and metrics independent of Ultralytics and file output."""

from dataclasses import dataclass
from pathlib import Path

from fruit_harvest.contract import Detection
from fruit_harvest.taxonomy import CLASS_NAMES


Box = tuple[float, float, float, float]
Truth = tuple[str, Box]


@dataclass(frozen=True)
class MatchRecord:
    """One image's truth and predictions with a unique spatial assignment."""

    truths: list[Truth]
    predictions: tuple[Detection, ...]
    pairs: list[tuple[int, int]]
    matched_truth: set[int]
    matched_prediction: set[int]


def ground_truth_for(image: Path, width: int, height: int) -> list[Truth]:
    """Convert this image's normalized YOLO labels back to pixel boxes."""
    label = image.parent.parent.parent / "labels" / image.parent.name / f"{image.stem}.txt"
    truths = []
    for line in label.read_text(encoding="utf-8").splitlines():
        class_id, cx, cy, box_width, box_height = line.split()
        cx, cy, box_width, box_height = map(float, (cx, cy, box_width, box_height))
        truths.append((
            CLASS_NAMES[int(class_id)],
            ((cx - box_width / 2) * width, (cy - box_height / 2) * height,
             (cx + box_width / 2) * width, (cy + box_height / 2) * height),
        ))
    return truths


def box_iou(first: Box, second: Box) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union else 0.0


def match_detections(truths: list[Truth], predictions: tuple[Detection, ...], minimum_iou: float) -> MatchRecord:
    """Greedily pair boxes by highest IoU, regardless of predicted class."""
    candidates = [
        (box_iou(truth_box, prediction.bbox_xyxy), truth_index, pred_index)
        for truth_index, (_, truth_box) in enumerate(truths)
        for pred_index, prediction in enumerate(predictions)
    ]
    pairs = []
    matched_truth: set[int] = set()
    matched_prediction: set[int] = set()
    for overlap, truth_index, pred_index in sorted(candidates, reverse=True):
        if overlap < minimum_iou:
            break
        if truth_index not in matched_truth and pred_index not in matched_prediction:
            pairs.append((truth_index, pred_index))
            matched_truth.add(truth_index)
            matched_prediction.add(pred_index)
    return MatchRecord(truths, predictions, pairs, matched_truth, matched_prediction)


def harvest_counts(records: list[MatchRecord], threshold: float) -> dict[str, float | int]:
    """Count ripe decisions; unmatched ripe predictions are false harvests."""
    tp = fp = fn = 0
    for record in records:
        matched_by_truth = dict(record.pairs)
        for truth_index, (truth_class, _) in enumerate(record.truths):
            pred_index = matched_by_truth.get(truth_index)
            predicted_harvest = (
                pred_index is not None
                and record.predictions[pred_index].class_name == "ripe"
                and record.predictions[pred_index].confidence >= threshold
            )
            if truth_class == "ripe":
                tp += int(predicted_harvest)
                fn += int(not predicted_harvest)
            elif predicted_harvest:
                fp += 1
        fp += sum(
            prediction.class_name == "ripe" and prediction.confidence >= threshold
            for index, prediction in enumerate(record.predictions)
            if index not in record.matched_prediction
        )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
