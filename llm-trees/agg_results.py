from pathlib import Path
import pandas as pd


def aggregate_results(
    predictions_root="results/predictions",
    method=None,
    output_csv="results/aggregated_metrics.csv",
):
    predictions_root = Path(predictions_root)

    rows = []

    if not predictions_root.exists():
        raise FileNotFoundError(f"Directory not found: {predictions_root}")

    for dataset_dir in predictions_root.iterdir():
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name

        method_dirs = []
        if method is None:
            method_dirs = [p for p in dataset_dir.iterdir() if p.is_dir()]
        else:
            method_dir = dataset_dir / method
            if method_dir.exists() and method_dir.is_dir():
                method_dirs = [method_dir]

        for method_dir in method_dirs:
            method_name = method_dir.name

            for seed_dir in method_dir.iterdir():
                if not seed_dir.is_dir():
                    continue

                metrics_file = seed_dir / "metrics.csv"
                if not metrics_file.exists():
                    print(f"Skipping missing file: {metrics_file}")
                    continue

                try:
                    df = pd.read_csv(metrics_file)

                    if df.empty:
                        print(f"Skipping empty file: {metrics_file}")
                        continue

                    row = df.iloc[0].to_dict()
                    row["dataset"] = dataset_name
                    row["method"] = method_name
                    row["seed_dir"] = seed_dir.name
                    rows.append(row)

                except Exception as e:
                    print(f"Failed to read {metrics_file}: {e}")

    if not rows:
        raise ValueError("No metrics.csv files were found.")

    all_results = pd.DataFrame(rows)

    if "roc_auc" not in all_results.columns:
        raise ValueError("Column 'roc_auc' not found in metrics.csv files.")

    if "f1_score" not in all_results.columns:
        raise ValueError("Column 'f1_score' not found in metrics.csv files.")

    if "acc" not in all_results.columns:
        raise ValueError("Column 'acc' not found in metrics.csv files.")

    invalid_mask = (
        (all_results["acc"] < 0)
        | (all_results["f1_score"] < 0)
        | (all_results["roc_auc"] < 0)
    )

    assert not invalid_mask.any(), (
        "Found unsuccessful experiments with negative metrics:\n"
        + all_results.loc[
            invalid_mask,
            ["dataset", "method", "seed_dir", "acc", "f1_score", "roc_auc"]
        ].to_string(index=False)
    )

    all_results["roc_auc"] = all_results["roc_auc"].apply(
        lambda x: max(x, 1 - x) if pd.notna(x) else x
    )

    grouped = (
        all_results
        .groupby("dataset", as_index=False)
        .agg(
            f1_score_avg=("f1_score", "mean"),
            f1_score_std=("f1_score", "std"),
            rocauc_avg=("roc_auc", "mean"),
            rocauc_std=("roc_auc", "std"),
        )
        .sort_values("dataset")
    )

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_path, index=False)

    print(f"Saved aggregated results to: {output_path}")
    print(grouped)


if __name__ == "__main__":
    aggregate_results(
        predictions_root="results/predictions",
        method="gpt-4o-mini",
        output_csv="results/aggregated_metrics.csv",
    )
