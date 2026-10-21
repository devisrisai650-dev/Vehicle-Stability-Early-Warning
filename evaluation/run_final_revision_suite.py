"""Run the repository's available final revision analyses in one command.

From the repository root:
    python .\evaluation\run_final_revision_suite.py

This script does not regenerate data, train LSTM/GRU, or benchmark hardware.
It runs the repeated-seed XGBoost/Hybrid regression ablation, surface-stratified
and run-cluster bootstrap analysis, current event-level evaluation, matched
validation-calibrated event-warning ablation, and then pytest. It stops on the
first failed stage so downstream outputs are not mistaken for a complete run.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "data/raw/retrofit_sensor_stream.csv",
    "data/raw/ground_truth_dynamics.csv",
    "data/raw/run_metadata.csv",
    "data/processed/train_sequences.npz",
    "data/processed/validation_sequences.npz",
    "data/processed/test_sequences.npz",
    "data/processed/predictions/xgboost_test_predictions.npy",
    "data/processed/predictions/lstm_test_predictions.npy",
    "data/processed/predictions/gru_test_predictions.npy",
    "data/processed/predictions/hybrid_test_predictions.npy",
    "data/processed/predictions/xgboost_validation_predictions.npy",
    "data/processed/predictions/hybrid_validation_predictions.npy",
    "data/processed/selected_warning_thresholds.npz",
]


def preflight() -> list[Path]:
    missing = [ROOT / item for item in REQUIRED_FILES if not (ROOT / item).is_file()]
    if missing:
        print("PREFLIGHT FAILED: required local artifacts are missing:", file=sys.stderr)
        for path in missing:
            print(f"  - {path.relative_to(ROOT)}", file=sys.stderr)
        print(
            "\nGenerate or restore these artifacts using the repository's existing "
            "data/model pipeline, then rerun this command. This script will not "
            "invent missing predictions or thresholds.",
            file=sys.stderr,
        )
        return missing
    return []


def run_stage(label: str, command: list[str]) -> None:
    print(f"\n{'=' * 78}\nSTAGE: {label}\nCOMMAND: {' '.join(command)}\n{'=' * 78}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    if preflight():
        return 2

    python = sys.executable
    stages = [
        (
            "Five-seed XGBoost / Hybrid regression ablation",
            [python, str(ROOT / "evaluation" / "repeated_seed_ablation_experiments.py")],
        ),
        (
            "Surface-stratified forecasting and run-cluster bootstrap",
            [python, str(ROOT / "evaluation" / "reviewer_revision_experiments.py")],
        ),
        (
            "Current event-level metrics for the four trained forecast models",
            [python, str(ROOT / "evaluation" / "event_lead_time.py")],
        ),
        (
            "Validation-calibrated event-warning ablation",
            [python, str(ROOT / "evaluation" / "event_warning_ablation.py")],
        ),
        ("Repository tests", [python, "-m", "pytest", "-q"]),
    ]

    completed = []
    for label, command in stages:
        try:
            run_stage(label, command)
        except subprocess.CalledProcessError as exc:
            print(
                f"\nSTOPPED: stage '{label}' exited with code {exc.returncode}. "
                "Review the error above, fix it, and rerun the suite. Completed "
                f"stages: {', '.join(completed) if completed else 'none'}.",
                file=sys.stderr,
            )
            return int(exc.returncode or 1)
        completed.append(label)

    print("\nALL REQUESTED LOCAL SUITE STAGES COMPLETED.")
    print("Review CSVs under results/tables/ and results/tables/reviewer_revision/.")
    print(
        "Scope reminder: this suite does not retrain LSTM/GRU across seeds, "
        "measure target-hardware latency/memory, calibrate sensor noise against "
        "hardware, or establish real-vehicle safety."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
