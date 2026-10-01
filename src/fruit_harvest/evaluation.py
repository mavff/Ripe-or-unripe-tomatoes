"""Held-out evaluation, spatial matching and harvest-threshold selection."""

import json
from collections import Counter
from pathlib import Path

import yaml

from fruit_harvest.decision import HarvestPolicy
from fruit_harvest.metrics import MatchRecord, ground_truth_for, harvest_counts, match_detections
from fruit_harvest.model import load_detector, predict_image, validate_detector
from fruit_harvest.taxonomy import CLASS_NAMES


def evaluate_model(
    weights: Path,
    dataset_yaml: Path,
    split: str,
    output: Path,
    image_size: int,
    min_confidence: float,
    match_iou: float,
    policy: HarvestPolicy | None = None,
    detection_only: bool = False,
) -> dict:
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be val or test")
    if split == "test" and policy is None and not detection_only:
        raise ValueError("Test requires a validation-selected policy or detection_only=True")
    if detection_only and (split != "test" or policy is not None):
        raise ValueError("Detection-only mode requires test split without a policy")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    descriptor = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    root = Path(descriptor["path"])
    image_dir = root / descriptor[split]
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise ValueError(f"No images in {image_dir}")
    detector = load_detector(weights)
    detector_metrics, per_class_detection = validate_detector(
        detector, dataset_yaml, split, image_size, output
    )
    records: list[MatchRecord] = []
    confusion = {truth: Counter() for truth in CLASS_NAMES}
    ground_truth_counts = Counter()
    unmatched_prediction_counts = Counter()
    errors = []
    for image in images:
        predicted = predict_image(detector, image, HarvestPolicy(), image_size, min_confidence)
        truths = ground_truth_for(image, predicted.width, predicted.height)
        ground_truth_counts.update(truth_class for truth_class, _ in truths)
        record = match_detections(truths, predicted.detections, match_iou)
        records.append(record)
        for truth_index, pred_index in record.pairs:
            truth_class = truths[truth_index][0]
            predicted_class = predicted.detections[pred_index].class_name
            confusion[truth_class][predicted_class] += 1
            if truth_class != predicted_class:
                errors.append({
                    "image": image.name, "truth": truth_class,
                    "truth_bbox_xyxy": truths[truth_index][1],
                    "prediction": predicted_class,
                    "prediction_bbox_xyxy": predicted.detections[pred_index].bbox_xyxy,
                    "confidence": predicted.detections[pred_index].confidence,
                })
        for truth_index, (truth_class, truth_box) in enumerate(truths):
            if truth_index not in record.matched_truth:
                errors.append({"image": image.name, "truth": truth_class, "truth_bbox_xyxy": truth_box, "prediction": "missed"})
        for pred_index, prediction in enumerate(predicted.detections):
            if pred_index not in record.matched_prediction:
                unmatched_prediction_counts[prediction.class_name] += 1
                errors.append({
                    "image": image.name, "truth": "none", "prediction": prediction.class_name,
                    "prediction_bbox_xyxy": prediction.bbox_xyxy,
                    "confidence": prediction.confidence,
                })
    policy_status = "provided" if policy else "detection_only" if detection_only else "selected_on_validation"
    if policy is None and split == "val":
        candidates = [round(value / 100, 2) for value in range(10, 96, 5)]
        threshold = max(
            candidates,
            key=lambda value: (
                harvest_counts(records, value)["f1"],
                harvest_counts(records, value)["precision"],
                value,
            ),
        )
        if harvest_counts(records, threshold)["tp"] == 0:
            policy_status = "unavailable_no_ripe_true_positives"
        else:
            policy = HarvestPolicy(threshold)
            policy.save(output / "policy.json")
    class_scores = {}
    for class_name in CLASS_NAMES:
        tp = confusion[class_name][class_name]
        support = sum(confusion[class_name].values())
        predicted_count = sum(confusion[truth][class_name] for truth in CLASS_NAMES)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        class_scores[class_name] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }
    matched_objects = sum(sum(counts.values()) for counts in confusion.values())
    correct_classes = sum(confusion[class_name][class_name] for class_name in CLASS_NAMES)
    report = {
        "schema_version": 1,
        "split": split,
        "images": len(images),
        "weights": str(weights.resolve()),
        "matching_iou": match_iou,
        "detector_metrics": detector_metrics,
        "per_class_detection": per_class_detection,
        "matched_class_confusion": {
            truth: {predicted: confusion[truth][predicted] for predicted in CLASS_NAMES}
            for truth in CLASS_NAMES
        },
        "ground_truth_objects": {name: ground_truth_counts[name] for name in CLASS_NAMES},
        "unmatched_predictions": {name: unmatched_prediction_counts[name] for name in CLASS_NAMES},
        "matched_objects": matched_objects,
        "matched_classification": {
            "accuracy": correct_classes / matched_objects if matched_objects else 0.0,
            "per_class": class_scores,
            "macro_f1": sum(score["f1"] for score in class_scores.values()) / len(CLASS_NAMES),
        },
        "policy_status": policy_status,
        "harvest_threshold": policy.ripe_threshold if policy else None,
        "harvest_decision": harvest_counts(records, policy.ripe_threshold) if policy else None,
        "error_examples": errors[:30],
        "total_errors": len(errors),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
