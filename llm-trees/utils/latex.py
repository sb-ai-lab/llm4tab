import pandas as pd

from results import calculate_baseline_diff


def generate_latex_table(approach: str, score: str, split: str, aggregation: str):

    datasets = [
        "boxing1",
        "boxing2",
        "japansolvent",
        "colic",
        "heart_h",
        "hepatitis",
        "house_votes_84",
        "labor",
        "penguins",
        "vote",
        "bankruptcy",
        "creditscore",
        "irish",
        "acl",
        "posttrauma"
    ]

    if approach == "induction":
        eval_results = pd.DataFrame()
        for sub_approach in ["induction", "optimal_test", "automl_test"]:
            eval_results_single = pd.read_csv(f"../results/{sub_approach}_results.csv")
            eval_results = pd.concat([eval_results, eval_results_single], ignore_index=True)
    elif approach == "embedding":
        eval_results = pd.read_csv(f"../results/{approach}_results.csv")

    # Group by dataset and method
    summary = eval_results[eval_results["split"] == split] \
        .groupby(["dataset", "method"]) \
        .agg({score: aggregation}) \
        .reset_index()
    summary = summary.pivot(
        index="dataset",
        columns="method",
        values=score) \
            .reindex(datasets) \
            .reset_index()

    if approach == "embedding":
        base_column = "no"
        summary, columns = calculate_baseline_diff(summary, base_column)

    # Add medians
    summary = pd.concat([
        summary.iloc[:10],
        pd.concat([pd.Series({"dataset": "median"}), summary.iloc[:10, 1:].median()]).to_frame().T,
        summary.iloc[10:13],
        pd.concat([pd.Series({"dataset": "median"}), summary.iloc[10:13, 1:].median()]).to_frame().T,
        summary.iloc[13:],
        pd.concat([pd.Series({"dataset": "median"}), summary.iloc[13:, 1:].median()]).to_frame().T
    ], ignore_index=True)

    # Formatting for LaTeX
    if approach == "induction":
        summary = summary.map(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
        columns = ["claude", "gemini", "gpt-4o", "gpt-o1", "bss", "oct", "autogluon",
                   "autoprognosis", "tabpfn"]
    elif approach == "embedding":
        summary[base_column] = summary[base_column].apply(lambda x: f"{x:.2f}")
        for col in columns:
            summary[col] = summary[col].apply(lambda x: f"{x:+.2f}")
        columns = ["no", "claude", "gemini", "gpt-4o", "gpt-o1", "rt-us", "et-ss",
                   "rf-ss", "xg-ss", "et-sv", "rf-sv", "xg-sv"]

    # LaTeX table rows
    rows = r"\textbf{Dataset}""" + "".join([f" & \\textbf{{{col}}}" for col in columns]) + \
        r"\\" + "\n"
    
    for _, row in summary.iterrows():
        dataset = row["dataset"]
        diffs = " & ".join(row[col] for col in columns)
        rows += f"{dataset} & {diffs} \\\\\n"


    return rows





if __name__ == "__main__":

    # Generate LaTeX tables
    for approach in ["induction", "embedding"]:
        for score in ["f1_score", "accuracy"]:
            latex_table = generate_latex_table(
                approach,
                score,
                "67/33",
                "median")
            print(latex_table)
            print("\n")
