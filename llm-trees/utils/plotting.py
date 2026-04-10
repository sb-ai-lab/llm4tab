import os

import matplotlib as mpl
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from llm_trees.config import Config
from llm_trees.utils import get_feature_count


def save_plot(approach: str, score: str, split: str, dataset: str, plot_type: str):

    sns.set_theme(style="whitegrid")
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.serif"] = "Helvetica"
    pylab.rcParams.update({
        "figure.figsize": (13, 6),
        "legend.fontsize": "x-large",
        "axes.labelsize": "xx-large",
        "axes.titlesize": "xx-large",
        "xtick.labelsize": "xx-large",
        "ytick.labelsize": "xx-large"
    })

    method_names = {
        "no": "No embedding",
        "claude": "Claude 3.5 Sonnet\nunsupervised",
        "gemini": "Gemini 1.5 Pro\nunsupervised",
        "gpt-4o": "GPT-4o\nunsupervised",
        "gpt-o1": "GPT-o1\nunsupervised",
        "bss": "BSS\nsupervised",
        "oct": "OCTs\nsupervised",
        "autogluon": "AutoGluon\nsupervised",
        "autoprognosis": "AutoPrognosis\nsupervised",
        "tabpfn": "TabPFN\nsupervised",
        "rt-us": "Random trees\nunsupervised",
        "et-ss": "Extra trees\nself-supervised",
        "rf-ss": "Random forest\nself-supervised",
        "xg-ss": "XGBoost\nself-supervised",
        "et-sv": "Extra trees\nsupervised",
        "rf-sv": "Random forest\nsupervised",
        "xg-sv": "XGBoost\nsupervised"
    }

    palette = {
        "Claude 3.5 Sonnet\nunsupervised": "tab:blue",
        "Gemini 1.5 Pro\nunsupervised": "tab:orange",
        "GPT-4o\nunsupervised": "tab:green",
        "GPT-o1\nunsupervised": "tab:red",
        "BSS\nsupervised": "tab:purple",
        "OCTs\nsupervised": "tab:purple",
        "AutoGluon\nsupervised": "tab:brown",
        "AutoPrognosis\nsupervised": "tab:brown",
        "TabPFN\nsupervised": "tab:pink",
        "No embedding": "tab:gray",
        "Random trees\nunsupervised": "tab:purple",
        "Extra trees\nsupervised": "tab:brown",
        "Random forest\nsupervised": "tab:brown",
        "XGBoost\nsupervised": "tab:brown",
        "Extra trees\nself-supervised": "tab:brown",
        "Random forest\nself-supervised": "tab:brown",
        "XGBoost\nself-supervised": "tab:brown",
        "Extra trees\nsupervised": "tab:pink",
        "Random forest\nsupervised": "tab:pink",
        "XGBoost\nsupervised": "tab:pink",
    }

    if dataset == "acl":
        hatches = 12 * [""] + 3 *["//"] + 3 * ["\\"] + \
            3 *["//"] + 2 * ["\\"] + 3 * [""] + \
            ["", "", "", "", "//", "\\", "//", "\\", ""]
    else:
        hatches = 12 * [""] + 3 *["//"] + 3 * ["\\"] + \
            3 *["//"] + 3 * ["\\"] + 3 * [""] + \
            ["", "", "", "", "//", "\\", "//", "\\", ""]


    if approach == "induction":
        eval_results = pd.DataFrame()
        for sub_approach in ["induction", "optimal_test", "automl_test"]:
            eval_results_single = pd.read_csv(f"results/{sub_approach}_results.csv")
            eval_results = pd.concat([eval_results, eval_results_single], ignore_index=True)
    elif approach == "embedding":
        eval_results = pd.read_csv(f"results/{approach}_results.csv")


    if plot_type == "boxplot":
        if dataset == "public":
            data = eval_results[(eval_results["split"] == split) & (eval_results["dataset"] != "acl") & (
                        eval_results["dataset"] != "posttrauma")]
        else:
            data = eval_results[(eval_results["split"] == split) & (eval_results["dataset"] == dataset)]

        data.loc[:, "method"] = data["method"].map(method_names)

        plt.figure()
        ax = sns.boxplot(x="method", y=score, hue="method", data=data, showfliers=False, palette=palette,
                         legend=False)
        plt.xticks(rotation=90, ha="right", va="center", rotation_mode="anchor")
        plt.xlabel("")
        plt.yticks(np.arange(0.0, 1.1, 0.1))
        plt.tight_layout()
        if score == "f1_score":
            plt.ylabel("F1-score")
            for file_format in ["pdf", "svg", "eps"]:
                plt.savefig(f"plots/{plot_type}_{approach}_{dataset}_{score}.{file_format}",
                            format=file_format, dpi=800)
        elif score == "accuracy":
            plt.ylabel("Balanced accuracy")
            for file_format in ["pdf", "svg", "eps"]:
                plt.savefig(f"plots/{plot_type}_{approach}_{dataset}_acc.{file_format}",
                            format=file_format, dpi=800)

    elif plot_type == "grouped_boxplot" and approach != "embedding":
        if dataset == "public":
            data = eval_results[(eval_results["dataset"] != "acl") & (eval_results["dataset"] != "posttrauma")]
        else:
            data = eval_results[eval_results["dataset"] == dataset]
        data.loc[:, "method"] = data["method"].map(method_names)

        plt.figure()
        ax = sns.boxplot(x="split", y=score, hue="method", data=data, showfliers=False, palette=palette)
        for i, patch in enumerate(ax.patches):
            patch.set_hatch(hatches[i])
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), shadow=True)
        plt.xlabel("")
        plt.yticks(np.arange(0.0, 1.1, 0.1))
        plt.tight_layout()
        if score == "f1_score":
            plt.ylabel("F1-score")
            for file_format in ["pdf", "svg", "eps"]:
                plt.savefig(f"plots/{plot_type}_{approach}_{dataset}_f1.{file_format}",
                            format=file_format, dpi=800)
        elif score == "accuracy":
            plt.ylabel("Balanced accuracy")
            for file_format in ["pdf", "svg", "eps"]:
                plt.savefig(f"plots/{plot_type}_{approach}_{dataset}_acc.{file_format}",
                            format=file_format, dpi=800)


def feature_count_analysis(dataset: str = "heart_h", num_trees: int = 5, root:str = "./"):

    config = Config(
        dataset=dataset,
        max_tree_depth=0,
        num_trees=num_trees,
        root=root,
    )

    filename = f"key_count_{config.dataset}_{config.num_trees}.png"
    folder_path = f"../plots/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {os.path.abspath(folder_path)}")
    save_to = os.path.join(folder_path, filename)

    feature_counts = get_feature_count(config)
    plot_key_counts(feature_counts, save_to)


def plot_key_counts(feature_counts, save_to):

    # Generate the index based on the keys of any model
    first_model = next(iter(feature_counts))
    keys = list(feature_counts[first_model].keys())
    index = np.arange(len(keys))  # Uniform index for the x-axis

    plt.figure(figsize=(12, 5))

    # Plot bars
    for idx, model in enumerate(feature_counts):
        feature_count_dict = feature_counts[model]
        values = [feature_count_dict[key] for key in keys]  # Ensure order matches keys

        bar_width = 0.8 / len(feature_counts)
        offset = 0.2 * idx  # Adjust bar positions to avoid overlap

        plt.bar(index - 0.4 + offset, values, bar_width, label=model, zorder=2)

    # Change xticks to human-readable format
    keys = [key.replace("_", " ") for key in keys]

    # Set xticks once using the consistent keys
    plt.xticks(index, keys, rotation="vertical")

    # Add labels, legend, and grid
    ax = plt.gca()
    plt.xlabel("")
    plt.ylabel("Count")
    plt.xticks(index, keys, rotation="vertical")
    plt.legend()
    ax.legend(loc='best', ncol=1)

    plt.savefig(save_to)
    print(f"Figure saved to: {save_to}")



def number_of_trees_vs_acc(result_path, dataset, x_param, y_metric, split="67/33", save_path=None):

    if not isinstance(result_path, list):
        result_path = [result_path]

    # Read results of all paths and concatenate the dataframes
    df = pd.concat([pd.read_csv(path) for path in result_path])

    # Filter for the desired split
    df = df[df["split"] == split]

    # Filter for the desired dataset
    df = df[df["dataset"] == dataset]

    # # Filter for no concatenation # TODO
    # df = df[df["concatenation"] == False]

    plt.figure(figsize=(12, 5))

    # Combine the best entry and the actual score into one column per method
    for method in df["method"].unique():

        df_filtered = df[df["method"] == method]
        df_filtered.groupby(x_param)[y_metric].mean().plot(label=method)
        plt.fill_between(
            df_filtered.groupby(x_param)[y_metric].mean().index,
            df_filtered.groupby(x_param)[y_metric].mean() - df_filtered.groupby(x_param)[y_metric].std(),
            df_filtered.groupby(x_param)[y_metric].mean() + df_filtered.groupby(x_param)[y_metric].std(),
            alpha=0.2,
        )

    plt.xticks(df[x_param].unique())
    plt.legend()
    plt.ylim(0, 1)
    plt.ylabel(y_metric.title())

    if save_path:
        plt.savefig(save_path, format="png", dpi=800)

    plt.show()




if __name__ == "__main__":

    for approach in ["induction", "embedding"]:
        for score in ["f1_score", "accuracy"]:
            for dataset in ["public", "acl", "posttrauma"]:
                for plot_type in ["boxplot", "grouped_boxplot"]:
                    save_plot(approach, score, "67/33", dataset, plot_type)
