import os
import json
import numpy as np
import pandas as pd

from .io import load_and_split_data
from .metrics import score_acc, score_f1, score_roc_auc
from .utils import get_tree_path, predict_tree_from_file, generate_tree, regenerate_tree


def get_prediction_result_dir(config):
    return os.path.join(
        config.root,
        "results",
        "predictions",
        config.dataset,
        config.method,
        f"seed_{config.iter}",
    )


def save_induction_outputs(config, y_true, y_pred, acc, f1, roc_auc):
    result_dir = get_prediction_result_dir(config)
    os.makedirs(result_dir, exist_ok=True)

    y_true_series = pd.Series(np.asarray(y_true).ravel(), name="y_true")
    y_true_series.to_csv(os.path.join(result_dir, "y_true.csv"), index=False)

    y_pred_series = pd.Series(np.asarray(y_pred).ravel(), name="y_pred")
    y_pred_series.to_csv(os.path.join(result_dir, "y_pred.csv"), index=False)

    metrics_df = pd.DataFrame([{
        "dataset": config.dataset,
        "method": config.method,
        "iter": config.iter,
        "seed": getattr(config, "seed", config.iter),
        "acc": acc,
        "f1_score": f1,
        "roc_auc": roc_auc,
    }])
    metrics_df.to_csv(os.path.join(result_dir, "metrics.csv"), index=False)

    metrics_payload = {
        "dataset": config.dataset,
        "method": config.method,
        "iter": config.iter,
        "seed": getattr(config, "seed", config.iter),
        "acc": float(acc) if acc == acc else None,
        "f1_score": float(f1) if f1 == f1 else None,
        "roc_auc": float(roc_auc) if roc_auc == roc_auc else None,
        "tree_path": get_tree_path(config),
    }
    with open(os.path.join(result_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)


def eval_induction(config):
    # try:
    X_train, X_test, y_train, y_test = load_and_split_data(config)

    # Generate tree if necessary
    generate_tree(config)

    unique_labels = np.unique(y_test)
    y_pred = None

    for try_idx in range(config.num_retry_llm + 1):
        if try_idx == config.num_retry_llm:
            raise ValueError(
                f"Reached maximum number of retries for tree: "
                f"{os.path.basename(get_tree_path(config))}"
            )

        try:
            y_pred, _ = predict_tree_from_file(config, X_test)
            if any(label not in unique_labels for label in np.unique(y_pred)):
                raise ValueError(f"Invalid labels in prediction: {np.unique(y_pred)}")
            break
        except Exception as e:
            regenerate_tree(config, e)

    if y_pred is None:
        raise ValueError(f"Prediction failed for tree: {get_tree_path(config)}")

    acc = score_acc(y_test, y_pred)
    f1 = score_f1(y_test, y_pred)
    roc_auc = score_roc_auc(y_test, y_pred)

    save_induction_outputs(config, y_test, y_pred, acc, f1, roc_auc)

    # except Exception as e:
    #     print(f"Scoring failed: {e}")
    #     acc, f1, roc_auc = -1, -1, -1

    return acc, f1, roc_auc