# Architecture and interfaces

```text
AgRob VOC, tomatOD COCO, or AerialYield YOLO + official COCO split
             │
             ▼
       Data preparation ──► YOLO images, labels, dataset.yaml, split manifest
             │
             ▼
      YOLO11n training ────► best.pt
             │
             ▼
     Candidate export ─────► model.onnx
             │
             ▼
 Validation evaluation ───► policy.json (selected on ONNX)
             │
             ├─────────────► held-out test report
             ▼
      Final packaging ─────► best.pt + same model.onnx + manifest.json
             │
             ▼
     Inference adapter ────► detections with class, box, confidence
             │
             ▼
      Harvest policy ──────► harvest or wait per detection
             │
             ▼
    Future arm controller
```

## Module boundaries

The `data/` package contains small source adapters (`coco.py`, `voc.py`, and `aerial.py`), verified downloads (`fetch.py`), shared annotation types (`types.py`), and dataset preparation (`prepare.py`). Each adapter translates its source labels to `unripe`, `semi_ripe`, and `ripe`; preparation checks images, boxes, and split overlap before writing a fixed YOLO layout. Splits are made by image, never by object. For AgRobTomato video frames, adjacent groups of 20 frames stay together. The original tomatOD test split is kept when supplied. For AerialYield, `aerial.py` reads the official COCO membership lists before selecting images from the YOLO export, whose folders contain identical images in multiple splits. The [split audit](../output/aerial/metrics/source_split_audit.json) records the discrepancy.

`model.py` is the only module that calls Ultralytics for training and prediction. It returns the project-owned `ImageResult` type. `config.py` checks training values before they reach Ultralytics. `metrics.py` contains spatial matching and harvest counts without model dependencies. `evaluation.py` orchestrates detector mAP measurement and selects a threshold on ONNX validation by F1; ties prefer precision and then the higher threshold. Test evaluation must load that saved policy.

`decision.py` has no dependency on PyTorch or image data. A future controller can consume the prediction JSON without loading the training code. The final bundle copies the exact ONNX candidate used for validation and test, and includes the preprocessing description, class order, threshold, file hashes, and schema version. `.pt` and `.onnx` use the same inference adapter and JSON contract, but their numeric scores can differ; ONNX is the calibrated deployment backend.

The diagram describes the intended deployment workflow. The published CPU experiments evaluate PyTorch checkpoints. An ONNX candidate needs its own validation and test results before using its scores for harvest decisions.

The scripts in `scripts/` do not affect inference. `audit_aerial_splits.py` records source integrity, `make_examples.py` draws labeled examples from held-out images, and `build_report.py` creates figures, public metric snapshots, and a PDF from saved experiment records. Training and evaluation parameters live in YAML files under `configs/`; each run has its own `args.yaml` and result CSV under ignored `artifacts/`.

## Prediction JSON contract

```json
{
  "schema_version": 1,
  "results": [
    {
      "schema_version": 1,
      "image": "path/to/tomato.jpg",
      "width": 1280,
      "height": 720,
      "detections": [
        {
          "bbox_xyxy": [100.0, 120.0, 180.0, 205.0],
          "class_name": "ripe",
          "confidence": 0.91,
          "decision": "harvest"
        }
      ]
    }
  ]
}
```

Coordinates are in original image pixels, from top-left to bottom-right. They identify a **fruit bounding box**, not a gripper pose or safe picking point. The arm integration will need camera calibration, depth or geometry, target selection, collision checks, and a separate actuation interface before any motion is possible.

## Planned educational comparison

A later custom CNN can classify cropped fruit boxes using the same image splits. Compare its maturity macro F1 and confusion matrix with the YOLO classifier on matched fruits. Report YOLO localization mAP separately; a crop classifier does not solve localization by itself.
