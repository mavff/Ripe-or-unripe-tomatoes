# Architecture and interfaces

```text
VOC or COCO annotations + images
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

The `data/` package contains small source adapters (`coco.py` and `voc.py`), the AgRob download helper (`fetch.py`), shared annotation types (`types.py`), and dataset preparation (`prepare.py`). The adapters translate source labels to `unripe`, `semi_ripe`, and `ripe`; preparation verifies boxes and images and writes a fixed YOLO layout. Splits are made by image, never by object. For AgRobTomato video frames, adjacent groups of 20 frames stay together to reduce leakage from near-duplicate images. The original tomatOD test split is kept when supplied.

`model.py` is the only module that calls Ultralytics for training and prediction. It returns the project-owned `ImageResult` type. `config.py` checks training values before they reach Ultralytics. `metrics.py` contains spatial matching and harvest counts without model dependencies. `evaluation.py` orchestrates detector mAP measurement and selects a threshold on ONNX validation by F1; ties prefer precision and then the higher threshold. Test evaluation must load that saved policy.

`decision.py` has no dependency on PyTorch or image data. A future controller can consume the prediction JSON without loading the training code. The final bundle copies the exact ONNX candidate used for validation and test, and includes the preprocessing description, class order, threshold, file hashes, and schema version. `.pt` and `.onnx` use the same inference adapter and JSON contract, but their numeric scores can differ; ONNX is the calibrated deployment backend.

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
