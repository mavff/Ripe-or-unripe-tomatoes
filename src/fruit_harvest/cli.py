"""Reproducible command-line workflow for data, training and inference."""

import argparse
import json
from pathlib import Path

from fruit_harvest.config import Settings
from fruit_harvest.data import fetch_agrob, prepare_dataset, read_coco, read_voc
from fruit_harvest.decision import HarvestPolicy
from fruit_harvest.evaluation import evaluate_model
from fruit_harvest.exporting import export_bundle, load_bundle
from fruit_harvest.model import load_detector, predict_image, train_model


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fruit-harvest", description="Tomato ripeness detection pipeline")
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="Prepare labeled tomato images")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    fetch = data_commands.add_parser("fetch-agrob", help="Download and verify AgRobTomato")
    fetch.add_argument("--output", type=_path, default=Path("data/raw"))
    prepare = data_commands.add_parser("prepare", help="Convert source annotations to YOLO")
    prepare.add_argument("--format", choices=("coco", "voc"), required=True)
    prepare.add_argument("--root", type=_path, help="Pascal VOC dataset root")
    prepare.add_argument("--images", type=_path, help="COCO source image directory")
    prepare.add_argument("--train-coco", type=_path)
    prepare.add_argument("--test-coco", type=_path)
    prepare.add_argument("--output", type=_path, default=Path("data/processed"))
    prepare.add_argument("--config", type=_path, default=Path("configs/default.yaml"))

    train = commands.add_parser("train", help="Fine-tune the detector")
    train.add_argument("--data", type=_path, default=Path("data/processed/dataset.yaml"))
    train.add_argument("--config", type=_path, default=Path("configs/default.yaml"))
    train.add_argument("--name", default="yolo11n-baseline")
    train.add_argument("--output", type=_path, default=Path("artifacts/runs"))

    evaluate = commands.add_parser("evaluate", help="Evaluate localization and harvest decisions")
    evaluate.add_argument("--weights", type=_path, required=True)
    evaluate.add_argument("--data", type=_path, default=Path("data/processed/dataset.yaml"))
    evaluate.add_argument("--config", type=_path, default=Path("configs/default.yaml"))
    evaluate.add_argument("--split", choices=("val", "test"), required=True)
    evaluate.add_argument("--policy", type=_path, help="Required for test; created by validation")
    evaluate.add_argument("--output", type=_path, required=True)

    export = commands.add_parser("export", help="Package PyTorch and ONNX model files")
    export.add_argument("--weights", type=_path, required=True)
    export.add_argument("--policy", type=_path, help="Validation-selected harvest policy")
    export.add_argument("--onnx", type=_path, help="Copy a previously evaluated ONNX candidate")
    export.add_argument("--config", type=_path, default=Path("configs/default.yaml"))
    export.add_argument("--output", type=_path, required=True)

    predict = commands.add_parser("predict", help="Return stable JSON for one image or a directory")
    predict.add_argument("--bundle", type=_path, required=True)
    predict.add_argument("--image", type=_path, required=True)
    predict.add_argument("--backend", choices=("pt", "onnx"), default="onnx")
    predict.add_argument("--output", type=_path, help="Write JSON here instead of stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "data":
        if args.data_command == "fetch-agrob":
            print(fetch_agrob(args.output))
            return 0
        settings = Settings.load(args.config)
        if args.format == "coco":
            if not all((args.images, args.train_coco, args.test_coco)):
                raise SystemExit("COCO requires --images, --train-coco and --test-coco")
            train_rows = read_coco(args.train_coco)
            test_rows = read_coco(args.test_coco)
            images_root = args.images
            source = "tomatOD COCO"
            validation_rows = None
        else:
            if args.root is None:
                raise SystemExit("VOC requires --root")
            train_rows, validation_rows, test_rows = read_voc(args.root, settings.seed)
            images_root = args.root
            source = "AgRobTomato Pascal VOC"
        result = prepare_dataset(
            images_root, train_rows, test_rows, args.output,
            settings.validation_fraction, settings.seed, source, validation_rows or None,
        )
        print(json.dumps(result["counts"], indent=2))
    elif args.command == "train":
        settings = Settings.load(args.config)
        print(train_model(args.data, settings, args.name, args.output))
    elif args.command == "evaluate":
        settings = Settings.load(args.config)
        policy = HarvestPolicy.load(args.policy) if args.policy else None
        result = evaluate_model(
            args.weights, args.data, args.split, args.output,
            settings.image_size, settings.prediction_confidence, settings.match_iou, policy,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "export":
        settings = Settings.load(args.config)
        if args.onnx and not args.policy:
            raise SystemExit("--onnx requires --policy for the final bundle")
        result = export_bundle(
            args.weights,
            HarvestPolicy.load(args.policy) if args.policy else None,
            args.output,
            settings.image_size,
            args.onnx,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "predict":
        weights, policy, image_size = load_bundle(args.bundle, args.backend)
        detector = load_detector(weights)
        if args.image.is_dir():
            images = sorted(
                path for path in args.image.iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
        else:
            images = [args.image]
        if not images:
            raise ValueError(f"No images found in {args.image}")
        result = {
            "schema_version": 1,
            "results": [predict_image(detector, image, policy, image_size).as_dict() for image in images],
        }
        serialized = json.dumps(result, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
        else:
            print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
