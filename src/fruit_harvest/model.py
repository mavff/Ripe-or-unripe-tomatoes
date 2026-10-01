"""Ultralytics adapter; other modules depend on project result types."""

from pathlib import Path

from fruit_harvest.config import Settings
from fruit_harvest.contract import Detection, ImageResult
from fruit_harvest.decision import HarvestPolicy
from fruit_harvest.taxonomy import CLASS_NAMES, canonical_class


def load_detector(weights: Path | str):
    from ultralytics import YOLO

    return YOLO(str(weights))


def detector_class_names(detector) -> list[str]:
    """Expose the model's class order through the project taxonomy."""
    return [canonical_class(detector.names[index]) for index in range(len(detector.names))]


def validate_detector(detector, dataset_yaml: Path, split: str, image_size: int,
                      output: Path) -> tuple[dict, dict]:
    """Return aggregate and per-class detection metrics from Ultralytics."""
    result = detector.val(
        data=str(dataset_yaml.resolve()), split=split, imgsz=image_size,
        device="cpu", workers=0, plots=True, project=str(output.resolve()),
        name="detector", exist_ok=True, verbose=False,
    )
    aggregate = {key: float(value) for key, value in result.results_dict.items()}
    per_class = {
        canonical_class(str(row["Class"])): {
            "images": int(row["Images"]),
            "instances": int(row["Instances"]),
            "precision": float(row["Box-P"]),
            "recall": float(row["Box-R"]),
            "f1": float(row["Box-F1"]),
            "map50": float(row["mAP50"]),
            "map50_95": float(row["mAP50-95"]),
        }
        for row in result.summary(decimals=8)
    }
    return aggregate, per_class


def export_detector_onnx(detector, image_size: int) -> Path:
    """Export the detector; bundle placement stays in the exporting module."""
    return Path(detector.export(
        format="onnx", imgsz=image_size, device="cpu", simplify=False, dynamic=False
    ))


def predict_image(
    detector,
    image: Path,
    policy: HarvestPolicy,
    image_size: int,
    min_confidence: float = 0.05,
) -> ImageResult:
    results = detector.predict(
        source=str(image), imgsz=image_size, conf=min_confidence, device="cpu", verbose=False
    )
    if len(results) != 1:
        raise RuntimeError(f"Expected one result for {image}, got {len(results)}")
    result = results[0]
    height, width = result.orig_shape
    detections = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        class_name = canonical_class(result.names[class_id])
        if class_name not in CLASS_NAMES:
            raise ValueError(f"Unexpected model class {class_name}")
        confidence = float(box.conf.item())
        coordinates = tuple(float(value) for value in box.xyxy[0].tolist())
        detections.append(
            Detection(coordinates, class_name, confidence, policy.decide(class_name, confidence))
        )
    return ImageResult(image, width, height, tuple(detections))


def train_model(dataset_yaml: Path, settings: Settings, run_name: str, project_dir: Path) -> Path:
    """Fine-tune a detector and return the best checkpoint from this run."""

    if (project_dir / run_name).exists():
        raise FileExistsError(f"Training run already exists: {project_dir / run_name}")
    detector = load_detector(settings.model)
    project_dir.mkdir(parents=True, exist_ok=True)
    detector.train(
        data=str(dataset_yaml.resolve()),
        imgsz=settings.image_size,
        batch=settings.batch_size,
        time=settings.max_hours,
        patience=settings.patience,
        workers=settings.workers,
        seed=settings.seed,
        optimizer=settings.optimizer,
        lr0=settings.initial_learning_rate,
        lrf=settings.final_lr_fraction,
        weight_decay=settings.weight_decay,
        cos_lr=settings.cosine_lr,
        cls_pw=settings.class_weight_power,
        momentum=0.9,
        warmup_bias_lr=0.0,
        device="cpu",
        project=str(project_dir.resolve()),
        name=run_name,
        exist_ok=False,
        cache=False,
        pretrained=True,
        plots=True,
    )
    checkpoint = project_dir / run_name / "weights" / "best.pt"
    if not checkpoint.exists():
        raise RuntimeError(f"Training finished without a best checkpoint at {checkpoint}")
    return checkpoint
