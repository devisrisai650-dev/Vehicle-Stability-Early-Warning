from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

GT_PATH = DATA_DIR / "ground_truth_dynamics.csv"
SENSOR_PATH = DATA_DIR / "retrofit_sensor_stream.csv"
META_PATH = DATA_DIR / "run_metadata.csv"


def main() -> None:

    gt = pd.read_csv(GT_PATH)
    sensor = pd.read_csv(SENSOR_PATH)
    meta = pd.read_csv(META_PATH)

    print("=" * 60)
    print("VEHICLE STABILITY DATASET AUDIT")
    print("=" * 60)

    print("\nGROUND TRUTH")
    print("-" * 60)
    print("Shape:", gt.shape)
    print("Columns:")
    print(gt.columns.tolist())

    print("\nRETROFIT SENSOR")
    print("-" * 60)
    print("Shape:", sensor.shape)
    print("Columns:")
    print(sensor.columns.tolist())

    print("\nRUN METADATA")
    print("-" * 60)
    print("Shape:", meta.shape)
    print("Columns:")
    print(meta.columns.tolist())

    print("\nMISSING VALUES")
    print("-" * 60)

    print("\nGround truth:")
    print(gt.isna().sum())

    print("\nSensor:")
    print(sensor.isna().sum())

    print("\nMetadata:")
    print(meta.isna().sum())

    print("\nINSTABILITY STATISTICS")
    print("-" * 60)

    print(
        "Total runs:",
        meta["Run_ID"].nunique()
    )

    print(
        "Runs reaching instability:",
        meta["reaches_instability"].sum()
    )

    print(
        "Instability percentage:",
        100 * meta["reaches_instability"].mean()
    )

    print("\nBy road surface:")
    print(
        meta.groupby("mu_surface")["reaches_instability"].mean()
        * 100
    )

    print("\nSTABILITY MARGIN")
    print("-" * 60)

    print(
        gt["stability_margin"].describe()
    )

    print("\nBETA")
    print("-" * 60)

    print(
        gt["beta"].describe()
    )

    print("\nTIME STEP")
    print("-" * 60)

    print(
        gt["Time"].diff().dropna().value_counts().head()
    )

    print("\nJOIN CHECK")
    print("-" * 60)

    gt_keys = set(zip(gt.Run_ID, gt.Time))
    sensor_keys = set(zip(sensor.Run_ID, sensor.Time))

    print("Ground-truth keys:", len(gt_keys))
    print("Sensor keys:", len(sensor_keys))
    print("Common keys:", len(gt_keys & sensor_keys))

    print("\nAUDIT COMPLETE")


if __name__ == "__main__":
    main()