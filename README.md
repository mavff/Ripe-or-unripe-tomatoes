# Fruit Harvest Vision

A small, reproducible computer vision project for detecting tomatoes on plants, estimating ripeness, and recommending whether to harvest each detected fruit. This repository covers the vision software. It does not move a robotic arm.

## What the system does

1. Converts annotated greenhouse images into a three-class YOLO dataset.
2. Fine-tunes a pretrained YOLO11 nano detector on CPU with AdamW, cosine learning-rate decay, and class-weighted classification loss.
3. Measures detection quality and selects a harvest confidence threshold on validation images.
4. Evaluates the selected threshold once on held-out test images.
5. Exports PyTorch weights, ONNX, and a manifest describing the inference contract.

The classes are `unripe`, `semi_ripe`, and `ripe`. The policy returns `harvest` only for a `ripe` detection at or above the selected threshold. All other detections return `wait`. A missing detection produces no harvest command.

## Hardware and setup

The default configuration targets a laptop with an Intel Iris Xe GPU, 15 GiB RAM, and CPU-only PyTorch. Training is capped at two hours by the Ultralytics `time` setting. That is a time budget, not a promised accuracy level. The source images contain small fruits; a larger input size may improve recall but will train more slowly.

```bash
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e .
source .venv/bin/activate
```

The dataset is not committed. See [Dataset](docs/dataset.md) for its source, licenses, labels, and preparation.

## Run the pipeline

For the AgRobTomato Pascal VOC archive, download, verify, extract, and prepare it with:

```bash
fruit-harvest data fetch-agrob

fruit-harvest data prepare --format voc \
  --root data/raw/Dataset-Greenhouse_Tomato_AgRob \
  --output data/agrob-processed

fruit-harvest train \
  --data data/agrob-processed/dataset.yaml \
  --name agrob-yolo11n

fruit-harvest export \
  --weights artifacts/runs/agrob-yolo11n/weights/best.pt \
  --output artifacts/model-candidate

fruit-harvest evaluate \
  --weights artifacts/model-candidate/model.onnx \
  --data data/agrob-processed/dataset.yaml \
  --split val --output artifacts/evaluation/validation

fruit-harvest evaluate \
  --weights artifacts/model-candidate/model.onnx \
  --data data/agrob-processed/dataset.yaml \
  --split test --policy artifacts/evaluation/validation/policy.json \
  --output artifacts/evaluation/test

fruit-harvest export \
  --weights artifacts/runs/agrob-yolo11n/weights/best.pt \
  --onnx artifacts/model-candidate/model.onnx \
  --policy artifacts/evaluation/validation/policy.json \
  --output artifacts/model-v1

fruit-harvest predict --bundle artifacts/model-v1 \
  --image path/to/tomato.jpg --output artifacts/prediction.json
```

Validation writes `policy.json` only after a correct `ripe` detection. If it reports `unavailable_no_ripe_true_positives`, the final export must wait for a better model or dataset.

Use `--backend pt` with `predict` to compare the PyTorch checkpoint to the default ONNX backend. The command also accepts a directory of `.jpg`, `.jpeg`, or `.png` images.

The original tomatOD annotations are COCO-compatible. When those files become available, prepare them with:

```bash
fruit-harvest data prepare --format coco \
  --images data/raw/tomatod/images \
  --train-coco data/raw/tomatod/train.json \
  --test-coco data/raw/tomatod/test.json \
  --output data/tomatod-processed
```

The COCO file names are examples; point the flags to the actual extracted files. The source test split is preserved, and validation is selected only from its training images.

## Source and dependency licenses

AgRobTomato is cited in [Dataset](docs/dataset.md); its Zenodo metadata does not display an explicit license, so images are kept local and are not redistributed here. The original tomatOD source states CC BY-NC-SA 4.0. Ultralytics distributes its software and models under AGPL-3.0 or a separate enterprise license; review those terms before deploying or distributing a derivative system. This repository does not commit datasets or trained weights.

## Repository map

| Path | Responsibility |
| --- | --- |
| `configs/default.yaml` | Training budget, optimization settings, split seed, and matching thresholds |
| `src/fruit_harvest/data/` | Source annotation adapters, validation, and YOLO conversion |
| `src/fruit_harvest/model.py` | Narrow Ultralytics adapter for training and prediction |
| `src/fruit_harvest/evaluation.py` | Validation and test orchestration, harvest policy selection |
| `src/fruit_harvest/metrics.py` | Spatial matching and harvest counts independent of model code |
| `src/fruit_harvest/decision.py` | Decision policy independent of model internals |
| `src/fruit_harvest/exporting.py` | Versioned `.pt` and `.onnx` bundle with checksums |
| `src/fruit_harvest/contract.py` | Stable result types for a future robot controller |
| `src/fruit_harvest/cli.py` | Reproducible user commands |
| `docs/` | Architecture, data provenance, and experiment guidance |

Read [Architecture](docs/architecture.md) for the contracts and [Experiments](docs/experiments.md) for metrics and interpretation.
