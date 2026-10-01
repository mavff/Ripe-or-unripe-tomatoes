"""Download, verify, and safely extract public source archives."""

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path


AGROB_URL = "https://zenodo.org/records/5596799/files/Dataset-Greenhouse_Tomato_AgRob.zip?download=1"
AGROB_MD5 = "890666716924415720f073b06a9a02a3"
AERIAL_BASE = "https://zenodo.org/records/22071809/files/"
AERIAL_ARCHIVES = {
    "aerial-coco.zip": ("dataset_detection_coco.zip", "60aba69c84331e0641ef98ec28130807"),
    "aerial-yolo.zip": ("dataset_detection_yolo.zip", "bc7f2fd258822a15ec580a8a81b85700"),
}


def _verified_archive(destination: Path, name: str, url: str, expected_md5: str) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / name
    if not archive.exists():
        temporary = destination / f"{name}.part"
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "fruit-harvest/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)

    digest = hashlib.md5()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_md5:
        raise ValueError(f"Archive checksum mismatch: {archive}")
    return archive


def _safe_extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        zipped.extractall(destination)


def fetch_agrob(destination: Path) -> Path:
    """Reuse a verified download or fetch and safely extract AgRobTomato."""
    archive = _verified_archive(destination, "agrob.zip", AGROB_URL, AGROB_MD5)
    root = destination / "Dataset-Greenhouse_Tomato_AgRob"
    if not root.exists():
        _safe_extract(archive, destination)
    if (len(list((root / "Annotations").glob("*.xml"))) != 449
            or len(list((root / "JPEGImages").glob("*.jpg"))) != 449):
        raise ValueError(f"Incomplete AgRobTomato extraction under {root}")
    return root


def fetch_aerial(destination: Path) -> Path:
    """Fetch YOLO images and the authoritative COCO split lists."""
    for local_name, (remote_name, checksum) in AERIAL_ARCHIVES.items():
        _verified_archive(destination, local_name, AERIAL_BASE + remote_name + "?download=1", checksum)
    root = destination / "aerial/dataset/detection/yolo"
    def complete() -> bool:
        return ((root / "tomato.yaml").is_file()
                and sum(1 for _ in (root / "images").rglob("*.jpg")) == 989
                and sum(1 for _ in (root / "labels").rglob("*.txt")) == 989)

    if not complete():
        _safe_extract(destination / "aerial-yolo.zip", destination / "aerial")
    if not complete():
        raise ValueError(f"Incomplete AerialYield extraction under {root}")
    return root
