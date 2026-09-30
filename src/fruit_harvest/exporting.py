"""Create a versioned, self-describing inference bundle."""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from fruit_harvest.decision import HarvestPolicy
from fruit_harvest.model import load_detector
from fruit_harvest.taxonomy import CLASS_NAMES, canonical_class


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_bundle(
    weights: Path,
    policy: HarvestPolicy | None,
    output: Path,
    image_size: int,
    onnx_source: Path | None = None,
) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Export directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "best.pt"
    shutil.copy2(weights, checkpoint)
    detector = load_detector(checkpoint)
    model_names = [canonical_class(detector.names[index]) for index in range(len(detector.names))]
    if model_names != list(CLASS_NAMES):
        raise ValueError(f"Model classes {model_names} do not match the export contract")
    onnx = output / "model.onnx"
    if onnx_source is not None:
        source_manifest = json.loads((onnx_source.parent / "manifest.json").read_text(encoding="utf-8"))
        if (source_manifest.get("schema_version") != 1
                or source_manifest.get("classes") != list(CLASS_NAMES)
                or source_manifest.get("image_size") != image_size
                or source_manifest["files"]["pytorch"]["sha256"] != _sha256(checkpoint)
                or source_manifest["files"]["onnx"]["sha256"] != _sha256(onnx_source)):
            raise ValueError("ONNX candidate does not match this checkpoint and model contract")
        shutil.copy2(onnx_source, onnx)
    else:
        exported = Path(
            detector.export(format="onnx", imgsz=image_size, device="cpu", simplify=False, dynamic=False)
        )
        exported.replace(onnx)
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_family": "YOLO11n",
        "task": "tomato_ripeness_detection",
        "classes": list(CLASS_NAMES),
        "image_size": image_size,
        "preprocessing": "Ultralytics letterbox, RGB, values normalized to [0, 1]",
        "bbox_format": "xyxy in original image pixels",
        "decision": (
            {"harvest_only_class": "ripe", "ripe_threshold": policy.ripe_threshold}
            if policy else None
        ),
        "files": {
            "pytorch": {"name": checkpoint.name, "sha256": _sha256(checkpoint)},
            "onnx": {"name": onnx.name, "sha256": _sha256(onnx)},
        },
        "dependencies": {
            "ultralytics": version("ultralytics"),
            "onnxruntime": version("onnxruntime"),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_bundle(bundle: Path, backend: str) -> tuple[Path, HarvestPolicy, int]:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("classes") != list(CLASS_NAMES):
        raise ValueError("Unsupported bundle contract")
    if manifest.get("decision") is None:
        raise ValueError("This candidate has no validated harvest policy")
    key = "pytorch" if backend == "pt" else "onnx"
    entry = manifest["files"][key]
    model_file = bundle / entry["name"]
    if _sha256(model_file) != entry["sha256"]:
        raise ValueError(f"Model checksum mismatch: {model_file}")
    return model_file, HarvestPolicy(float(manifest["decision"]["ripe_threshold"])), int(manifest["image_size"])
