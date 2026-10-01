"""Choose the AerialYield checkpoint from validation reports only."""

import json
from pathlib import Path


RUNS = {
    "320px, peso 0.25": ("aerial-first-validation", "aerial-adamw-cosine", "configs/aerial-cpu.yaml"),
    "320px, peso 0.75": ("aerial-classweight-validation", "aerial-adamw-classweight", "configs/aerial-cpu-classweight.yaml"),
    "512px, peso 0.50": ("aerial-highres-validation", "aerial-adamw-highres", "configs/aerial-cpu-highres.yaml"),
}


def summarize(report: dict) -> dict:
    ripe = report["matched_classification"]["per_class"]["ripe"]
    harvest = report["harvest_decision"]
    return {
        "policy_status": report["policy_status"],
        "detection_map50": report["detector_metrics"]["metrics/mAP50(B)"],
        "matched_macro_f1": report["matched_classification"]["macro_f1"],
        "matched_ripe_support": ripe["support"],
        "matched_ripe_precision": ripe["precision"],
        "matched_ripe_recall": ripe["recall"],
        "matched_ripe_f1": ripe["f1"],
        "validation_harvest_f1": harvest["f1"] if harvest else None,
        "validation_harvest_precision": harvest["precision"] if harvest else None,
    }


def selection_key(row: dict) -> tuple:
    """Favor a validated harvest policy, then ripe F1, macro F1, and mAP."""
    return (
        int(row["policy_status"] == "selected_on_validation"),
        row["validation_harvest_f1"] or 0.0,
        row["matched_ripe_f1"],
        row["matched_macro_f1"],
        row["detection_map50"],
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results = {}
    public_reports = root / "output/aerial/metrics/validation_runs"
    public_reports.mkdir(parents=True, exist_ok=True)
    public_histories = root / "output/aerial/metrics/training_runs"
    public_histories.mkdir(parents=True, exist_ok=True)
    for label, (directory, train_run, config) in RUNS.items():
        path = root / "artifacts/evaluation" / directory / "report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["split"] != "val":
            raise ValueError(f"Not a validation report: {path}")
        results[label] = {"train_run": train_run, "config": config, **summarize(report)}
        public_report = {**report, "weights": f"artifacts/runs/{train_run}/weights/best.pt"}
        (public_reports / f"{train_run}.json").write_text(
            json.dumps(public_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        source_history = root / "artifacts/runs" / train_run / "results.csv"
        (public_histories / f"{train_run}.csv").write_text(
            source_history.read_text(encoding="utf-8"), encoding="utf-8"
        )
    selected = max(results, key=lambda label: selection_key(results[label]))
    payload = {
        "selection_rule": "validated harvest policy; then validation harvest F1, matched ripe F1, matched macro F1, detection mAP50",
        "selected_run": selected,
        "selected_checkpoint": f"artifacts/runs/{results[selected]['train_run']}/weights/best.pt",
        "selected_config": results[selected]["config"],
        "runs": results,
        "test_set_used_for_selection": False,
    }
    path = root / "output/aerial/metrics/validation_comparison.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(path)
    print(f"Selected on validation: {selected}")


if __name__ == "__main__":
    main()
