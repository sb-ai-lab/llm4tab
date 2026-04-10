import csv
import os

import pandas as pd
from prettytable import PrettyTable

from llm_trees.config import Config


class ResultHandler:
    def __init__(self, result_path, fieldnames):
        self.result_path = result_path
        self.fieldnames = fieldnames
        self.existing_results = []
        if Config.skip_existing:
            self._load_existing_results()
        else:
            # delete existing results
            if os.path.exists(self.result_path):
                os.remove(self.result_path)

    def _load_existing_results(self):
        """Load existing results from the CSV file into a set."""
        if os.path.exists(self.result_path):
            with open(self.result_path, mode="r", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    self.existing_results.append(self.get_result_key_from_row(row))

    def log_result(self, config, accuracy, f1_score):
        """Log a new result to the CSV file."""

        # Do not log invalid runs
        if accuracy < 0:
            return

        result_key = self.get_result_key_from_config(config)
        if result_key in self.existing_results:
            return  # Skip duplicate results

        # Build the result dictionary
        result = self.build_result_dict(config, accuracy, f1_score)

        with open(self.result_path, mode="a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            if os.stat(self.result_path).st_size == 0:
                writer.writeheader()
            writer.writerow(result)
        self.existing_results.append(result_key)

    def is_result_present(self, config):
        """Check if a result already exists."""
        current_result_key = self.get_result_key_from_config(config)
        # print(current_result_key)
        return current_result_key in self.existing_results

    def build_result_dict(self, config, accuracy, f1_score):
        """Build the result dictionary for logging."""
        result = {
            "dataset": config.dataset,
            "method": config.method,
            "split": config.split_str(),
            "iter": config.iter,
            "accuracy": accuracy,
            "f1_score": f1_score,
        }
        # Add additional fields specific to the subclass
        result.update(self.additional_fields(config))
        return result

    def additional_fields(self, config):
        """Return additional fields specific to the subclass."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_result_key_from_config(self, config):
        """Return a unique identifier for a Config object."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_result_key_from_row(self, row):
        """Return a unique identifier for a row in the CSV file."""
        raise NotImplementedError("Subclasses must implement this method.")


class InductionAblationResultHandler(ResultHandler):
    def __init__(self, result_path):
        fieldnames = [
            "dataset",
            "method",
            "num_trees",
            "iter",
            "decision_tree",
            "description",
            "num_examples",
            "max_tree_depth",
            "temperature",
            "split",
            "accuracy",
            "f1_score",
        ]
        super().__init__(result_path, fieldnames)

    def additional_fields(self, config):
        """Return additional fields specific to the induction experiment."""
        return {
            "num_trees": config.num_trees,
            "decision_tree": config.force_decision_tree,
            "description": config.include_description,
            "num_examples": config.num_examples,
            "max_tree_depth": config.max_tree_depth,
            "temperature": config.temperature,
        }

    def get_result_key_from_config(self, config):
        """Return a unique identifier for a Config object."""

        return (
            config.dataset,
            config.method,
            config.num_trees,
            config.force_decision_tree,
            config.include_description,
            config.num_examples,
            config.max_tree_depth,
            int(100 * config.temperature),
            config.split_str(),
            config.iter,
        )

    def get_result_key_from_row(self, row):
        """Return a unique identifier for a row in the CSV file."""
        return (
            row["dataset"],
            row["method"],
            eval(row["num_trees"]),
            eval(row["decision_tree"]),
            eval(row["description"]),
            eval(row["num_examples"]),
            eval(row["max_tree_depth"]),
            int(100 * eval(row["temperature"])),
            row["split"],
            eval(row["iter"]),
        )

class EmbeddingAblationResultHandler(ResultHandler):
    def __init__(self, result_path):
        fieldnames = [
            "dataset",
            "method",
            "classifier",
            "concatenation",
            "num_trees",
            "num_examples",
            "description",
            "max_tree_depth",
            "temperature",
            "split",
            "iter",
            "accuracy",
            "f1_score",
        ]
        super().__init__(result_path, fieldnames)

    def additional_fields(self, config):
        """Return additional fields specific to the embedding experiment."""
        return {
            "classifier": config.classifier,
            "concatenation": config.append_raw_features,
            "num_trees": config.num_trees,
            "num_examples": config.num_examples,
            "description": config.include_description,
            "max_tree_depth": config.max_tree_depth,
            "temperature": config.temperature,
        }

    def get_result_key_from_config(self, config):
        """Return a unique identifier for a Config object."""

        return (
            config.dataset,
            config.method,
            config.classifier,
            config.append_raw_features,
            config.num_trees,
            config.num_examples,
            config.include_description,
            config.max_tree_depth,
            int(100 * config.temperature),
            config.split_str(),
            config.iter,
        )

    def get_result_key_from_row(self, row):
        """Return a unique identifier for a row in the CSV file."""
        return (
            row["dataset"],
            row["method"],
            row["classifier"],
            eval(row["concatenation"]),
            eval(row["num_trees"]),
            eval(row["num_examples"]),
            eval(row["description"]),
            eval(row["max_tree_depth"]),
            int(100 * eval(row["temperature"])),
            row["split"],
            eval(row["iter"]),
        )

class EmbeddingResultHandler(ResultHandler):
    def __init__(self, result_path):
        fieldnames = [
            "dataset",
            "method",
            "num_trees",
            "split",
            "iter",
            "accuracy",
            "f1_score",
        ]
        super().__init__(result_path, fieldnames)

    def additional_fields(self, config):
        """Return additional fields specific to the embedding experiment."""
        return {
            "num_trees": config.num_trees,
        }

    def get_result_key_from_config(self, config):
        """Return a unique identifier for a Config object."""
        return (
            config.dataset,
            config.method,
            config.num_trees,
            config.split_str(),
            config.iter,
        )

    def get_result_key_from_row(self, row):
        """Return a unique identifier for a row in the CSV file."""
        return (
            row["dataset"],
            row["method"],
            eval(row["num_trees"]),
            row["split"],
            eval(row["iter"]),
        )

class InductionResultHandler(ResultHandler):
    def __init__(self, result_path):
        fieldnames = [
            "dataset",
            "method",
            "iter",
            "split",
            "accuracy",
            "f1_score",
        ]
        super().__init__(result_path, fieldnames)

    def additional_fields(self, config):
        """Return additional fields specific to the induction experiment."""
        return {}

    def get_result_key_from_config(self, config):
        """Return a unique identifier for a Config object."""
        return (
            config.dataset,
            config.method,
            config.split_str(),
            config.iter,
        )

    def get_result_key_from_row(self, row):
        """Return a unique identifier for a row in the CSV file."""
        return (
            row["dataset"],
            row["method"],
            row["split"],
            eval(row["iter"]),
        )


def get_result_summary(result_path, title="Results", score="f1_score", split="67/33", aggregation="median", embedding_methods=None):
    if not isinstance(result_path, list):
        result_path = [result_path]

    # Read results of all pathes and concatenate the dataframes
    df = pd.concat([pd.read_csv(path) for path in result_path])

    # Filter for the desired split
    df = df[df["split"] == split]

    summary = df.groupby(["dataset", "method"]).agg({score: aggregation}).reset_index()

    # Calculate median values across all datasets for each metric/method
    median = df.groupby("method").agg({score: aggregation}).reset_index().set_index('method').T
    median.index = [aggregation.title()]

    pivot_summary = summary.pivot(index="dataset", columns="method", values=[score])
    pivot_summary.columns = [f"{method}" for metric, method in pivot_summary.columns]

    # rearrange columns, so that they are in the order of embedding_methods.keys()
    if "no" in pivot_summary.columns:
        pivot_summary = pivot_summary[embedding_methods]

    # Append median row to the summary
    pivot_summary = pd.concat([pivot_summary, median])

    if "no" in pivot_summary.columns:
        pivot_summary, columns = calculate_baseline_diff(pivot_summary, "no")

        for col in columns:
            pivot_summary[col] = pivot_summary[col].apply(lambda x: format_percentage(x, add_sign=True))


    # Pretty print the table
    table = PrettyTable()
    table.title = title
    table.field_names = ["Dataset"] + list(pivot_summary.columns)
    for index, row in pivot_summary.iterrows():
        row_data = [f"{value:.2f}" if isinstance(value, (float, int)) else value for value in row]

        if index == aggregation.title():
            table.add_row(["-" * 8] + ["-" * len(str(value)) for value in row_data])
        table.add_row([index] + row_data)

    # Save the table to a file
    table_txt = table.get_string()
    with open(result_path[0].replace(".csv", f"_{score}_summary.txt"), "w") as f:
        f.write(table_txt)

    return table

def calculate_baseline_diff(summary, base_column="no"):
    """
    Calculate the F1-score differences for each method relative to 'no embedding'.
    Returns a DataFrame with the results.
    """

    # Assuming "no embedding" column is included
    columns = list(summary.columns)
    columns.remove(base_column)

    if "dataset" in columns:
        columns.remove("dataset")

    # Calculate differences relative to "no embedding"
    for col in columns:
        summary[col] = summary[col] - summary[base_column]

    return summary, columns

def get_best_results(result_path, x_param, y_metric, split="67/33"):
    if not isinstance(result_path, list):
        result_path = [result_path]

    # Read results of all paths and concatenate the dataframes
    df = pd.concat([pd.read_csv(path) for path in result_path])

    # Filter for the desired split
    df = df[df["split"] == split]

    # Get the best y_metric over x_param for each dataset and method
    idx = df.groupby(["dataset", "method"])[y_metric].idxmax()
    best_results = df.loc[idx, ["dataset", "method", x_param, y_metric]]

    # Pivot the dataframe to have datasets as rows and methods as columns
    pivot_best_results = best_results.pivot(index="dataset", columns="method", values=[x_param, y_metric])
    pivot_best_results.columns = [f"{metric}_{method}" for metric, method in pivot_best_results.columns]

    # # get datframe with the most common best result
    # best_results_count = best_results.groupby(["dataset", "method"]).size().reset_index(name='Mode')
    # idx = best_results_count.groupby(["dataset"])["Mode"].idxmax()
    # best_results_mode = best_results_count.loc[idx, ["dataset", "method", "Mode"]]

    # Combine the best entry and the actual score into one column per method
    for method in df["method"].unique():
        pivot_best_results[method] = pivot_best_results.apply(
            lambda row: format_row(row, method, x_param, y_metric), axis=1
        )

    # Drop the separate x_param and y_metric columns
    pivot_best_results = pivot_best_results[[method for method in df["method"].unique()]]

    # Append median row to the summary
    # pivot_best_results = pd.concat([pivot_best_results, median])

    # # convert to pretty table
    # table = PrettyTable()
    # table.title = f"Best {y_metric.title()} over {x_param.title()}"
    # table.field_names = ["Dataset"] + list(pivot_best_results.columns)
    # for index, row in pivot_best_results.iterrows():
    #     table.add_row([index] + list(row))

    return pivot_best_results

def format_percentage(value, add_sign=False):
    if add_sign:
        sign = '+' if value > 0 else ''
        return f"{sign}{value:.2f}"
    else:
        return f"{value:.2f}"

def format_row(row, method, x_param, y_metric):
    x_val = row[f"{x_param}_{method}"]
    y_val = row[f"{y_metric}_{method}"]
    
    if isinstance(x_val, int):
        return f"{x_val:.0f} ({y_val:.2f})"
    elif isinstance(x_val, float):
        if x_val.is_integer():
            return f"{int(x_val)} ({y_val:.2f})"
        else:
            return f"{x_val:.1f} ({y_val:.2f})"
    else:
        return f"{x_val} ({y_val:.2f})"
