import pandas as pd


def compute_diffs(approach: str, split:str, aggregation: str, ablation: str):

    # dict for path and non-default values
    if approach == "induction":
        ablation_dict = {
            "max_tree_depth": [
                "../results/ablations/setting_1_optimal_tree_depth_induction.csv",
                [1, 3, 4, 5]
            ],
            "temperature": [
                "../results/ablations/setting_4_optimal_temperature_induction.csv",
                [0.5, 1]
            ],
            "num_examples": [
                "../results/ablations/setting_6_optimal_number_of_examples_induction.csv",
                [2, 3]
            ],
            "description": [
                "../results/ablations/setting_10_optimal_description_induction.csv",
                [True]
            ],
            "decision_tree": [
                "../results/ablations/setting_9_optimal_decision_tree_vs_free_form_induction.csv",
                [False]
            ]
        }
    elif approach == "embedding":
        ablation_dict = {
            "max_tree_depth": [
                "../results/ablations/setting_2_optimal_tree_depth_embeddings.csv",
                [1, 2, 3, 4, 5]
            ],
            "temperature": [
                "../results/ablations/setting_5_optimal_temperature_embeddings.csv",
                [0, 0.5]
            ],
            "num_examples": [
                "../results/ablations/setting_7_optimal_number_of_examples_embeddings.csv",
                [2, 3]
            ],
            "description": [
                "../results/ablations/setting_11_optimal_description_embeddings.csv",
                [True]
            ],
            "num_trees": [
                "../results/ablations/setting_3_optimal_number_of_trees_embeddings.csv",
                [1, 2, 3, 4]
            ],
            "concatenation": [
                "../results/ablations/setting_8_optimal_append_raw_features_embeddings.csv",
                [False]
            ],
            "classifier": [
                "../results/ablations/setting_12_optimal_classifier_embeddings.csv",
                ["hgbdt", "lr"]
            ]
        }
    
    default_results = pd.read_csv(f"../results/{approach}_results.csv")
    ablation_results = pd.read_csv(ablation_dict[ablation][0])

    scores = ["f1_score", "accuracy"]
    diffs = pd.DataFrame(columns = scores)
    for score in scores:
        # compute aggregate default score per method
        filter = (default_results["split"] == split) & (default_results["method"] \
                                                         .isin(["claude", "gemini", "gpt-4o", "gpt-o1"]))
        score_of_default = default_results[filter] \
            .groupby(["method"]) \
            .agg({score: aggregation})
        
        # compute best aggregate non-default score per method
        filter = (ablation_results["split"] == split) & (ablation_results[ablation] \
                                                         .isin(ablation_dict[ablation][1]))
        ablation_results_filtered = ablation_results[filter] \
            .groupby(["method", ablation]) \
            .agg({score: aggregation})
        best_score_of_non_default = ablation_results_filtered \
            .groupby("method") \
            .agg({score: "max"})
        
        # best aggregate non-default score - aggregate default score, i.e., positive values
        # indicate better non-default, negative values indicate better default
        diff = best_score_of_non_default - score_of_default
        diffs[score] = diff.round(2)

    diffs["combined"] = diffs["f1_score"].astype(str) + " / " + diffs["accuracy"].astype(str)

    return diffs["combined"].to_frame().T





if __name__ == "__main__":

    # print LaTeX table
    diffs = pd.DataFrame()
    for approach in ["induction", "embedding"]:
        if approach == "induction":
            ablations = [
                "max_tree_depth",
                "temperature",
                "num_examples",
                "description",
                "decision_tree"
            ]
        elif approach == "embedding":
            ablations = [
                "max_tree_depth",
                "temperature",
                "num_examples",
                "description",
                "num_trees",
                "concatenation",
                "classifier"
            ]
        for ablation in ablations:
            diff = compute_diffs(
                approach,
                "67/33",
                "median",
                ablation
            )
            diffs = pd.concat([diffs, diff], ignore_index=True)

    print(diffs.to_latex())
