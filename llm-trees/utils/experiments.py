import os

import pandas as pd
from tqdm import tqdm
import results
from llm_trees.config import Config
from llm_trees.embedding_utils import tree_to_embedding
from llm_trees.embeddings import eval_embedding
from llm_trees.induction_utils import eval_induction
from llm_trees.io import get_data
from llm_trees.utils import get_tree_path, count_keys_in_file

datasets = [
    "acl",
    "bankruptcy",
    "boxing1",
    "boxing2",
    "creditscore",
    "japansolvent",
    "colic",
    "heart_h",
    "hepatitis",
    "house_votes_84",
    "irish",
    "labor",
    "penguins",
    "posttrauma",
    "vote",
]

induction_methods = {
    "claude": "Claude 3.5 Sonnet",
    "gemini": "Gemini 1.5 Pro",
    "gpt-4o": "GPT-4o",
    "gpt-o1": "GPT-o1",
}

embedding_methods = {
    "no": "No Embedding",
    "claude": "Claude 3.5 Sonnet\nunsupervised",
    "gemini": "Gemini 1.5 Pro\nunsupervised",
    "gpt-4o": "GPT-4o\nunsupervised",
    "gpt-o1": "GPT-o1\nunsupervised",
    "rt-us": "Random Trees\nunsupervised",
    "et-ss": "Extra Trees\nself-supervised",
    "rf-ss": "Random Forest\nself-supervised",
    "xg-ss": "XGBoost\nself-supervised",
    "et-sv": "Extra Trees\nsupervised",
    "rf-sv": "Random Forest\nsupervised",
    "xg-sv": "XGBoost\nsupervised",
}

induction_splits = [
    0.33,
    0.5,
    0.67,
]

embedding_splits = [
    0.33,
    0.5,
    0.67,
]


def induction(
        is_ablation=True,
        force_decision_tree_list=None,
        max_tree_depth_list=None,
        num_examples_list=None,
        result_path="./results/optimization_results.csv",
        temperature_list=None,
        include_description_list = None,
        dataset_list = None,
        model_list = None,
):

    if model_list is None:
        model_list = induction_methods.keys()

    if dataset_list is None:
        dataset_list = datasets

    check_inputs(max_tree_depth_list, num_examples_list, result_path, temperature_list, include_description_list)

    if is_ablation:
        result_handler = results.InductionAblationResultHandler(result_path)
    else:
        result_handler = results.InductionResultHandler(result_path)

    for dataset_idx, dataset in enumerate(dataset_list):

        # Check if data is available
        X, y = get_data(os.getcwd(), dataset)
        if X is None or y is None:
            print(f"Data not found for dataset: {dataset}")
            continue

        for method_idx, method in tqdm(enumerate(model_list), f"{os.path.basename(result_path)}  [{dataset_idx}]: {dataset}"):
            for num_examples in num_examples_list:
                for max_tree_depth in max_tree_depth_list:
                    for temperature in temperature_list:
                        for force_decision_tree in force_decision_tree_list:
                            for include_description in include_description_list:
                                for split_idx, split in enumerate(induction_splits):

                                    temp = temperature[method] if isinstance(temperature, dict) else temperature
                                    config = Config(
                                        dataset=dataset,
                                        force_decision_tree=force_decision_tree,
                                        include_description=include_description,
                                        max_tree_depth=max_tree_depth,
                                        method=method,
                                        num_examples=num_examples,
                                        num_iters=1,
                                        root=os.getcwd(),
                                        temperature=temp,
                                        train_split=split,
                                    )

                                    for tree_idx in range(config.num_trees):
                                        config.iter = tree_idx
                                        config.seed = tree_idx

                                        if config.skip_existing and result_handler.is_result_present(config):
                                            continue  # Skip if result already exists

                                        acc, f1 = eval_induction(config)
                                        result_handler.log_result(config, acc, f1)



def embeddings(
        is_ablation=True,
        num_trees_list=None,
        num_examples_list=None,
        max_tree_depth_list=None,
        temperature_list=None,
        append_raw_features_list=None,
        include_description_list = None,
        result_path="./results/optimization_results.csv",
        dataset_list = None,
        model_list = None,
        classifier_list = None,
):

    if model_list is None:
        model_list = embedding_methods.keys()

    if dataset_list is None:
        dataset_list = datasets

    if classifier_list is None:
        classifier_list = ["mlp"]

    check_inputs(max_tree_depth_list, num_examples_list, result_path, temperature_list, include_description_list)

    if is_ablation:
        result_handler = results.EmbeddingAblationResultHandler(result_path)
    else:
        result_handler = results.EmbeddingResultHandler(result_path)

    for dataset_idx, dataset in enumerate(dataset_list):

        # Check if data is available
        X, y = get_data(os.getcwd(), dataset)
        if X is None or y is None:
            print(f"Data not found for dataset: {dataset}")
            continue

        for method in tqdm(model_list, f"{os.path.basename(result_path)} [{dataset_idx}]: {dataset}"):
            for num_trees in num_trees_list:
                for num_examples in num_examples_list:
                    for max_tree_depth in max_tree_depth_list:
                        for temperature in temperature_list:
                            for append_raw_features in append_raw_features_list:
                                for include_description in include_description_list:
                                    for classifier in classifier_list:
                                        for split in embedding_splits:

                                            config = Config(
                                                classifier=classifier,
                                                append_raw_features=append_raw_features,
                                                dataset=dataset,
                                                force_decision_tree=True,
                                                include_description=include_description,
                                                max_tree_depth=max_tree_depth,
                                                method=method,
                                                num_examples=num_examples,
                                                num_iters=5,
                                                num_trees=num_trees,
                                                root=os.getcwd(),
                                                train_split=split,
                                            )

                                            for iter in range(config.num_iters):
                                                config.iter = iter
                                                config.seed = iter

                                                if method in ["claude", "gemini", "gpt-4o", "gpt-o1", "gpt"]:
                                                    config.temperature = temperature[method] if isinstance(temperature, dict) else temperature

                                                if config.skip_existing and result_handler.is_result_present(config):
                                                    continue  # Skip if result already exists

                                                acc, f1 = eval_embedding(config)

                                                config.iter = iter
                                                result_handler.log_result(config, acc, f1)



def embedding_dimensions(
        result_path="./results/embedding_dimensions",
        dataset_list=None,
        model_list=None,
):

    if model_list is None:
        model_list = list(embedding_methods.keys())[1:5]
    if dataset_list is None:
        dataset_list = datasets

    embedding_dimensions = pd.DataFrame(
        data=0,
        index=dataset_list,
        columns=["Features set size"] + model_list,
        dtype=int,
    )
    num_features = pd.DataFrame(
        data=0,
        index=dataset_list,
        columns=["Features set size"] + model_list,
        dtype=int,
    )

    embedding_dimensions.index.name = 'Dataset'
    num_features.index.name = 'Dataset'

    for dataset in tqdm(dataset_list, "Calculating embedding_dimensions"):

        X, y = get_data(os.getcwd(), dataset)

        if X is None or y is None:
            print(f"Data not found for dataset: {dataset}")
            continue

        embedding_dimensions["Features set size"][dataset] = len(X.keys())
        num_features["Features set size"][dataset] = len(X.columns)

        for method in model_list:
            config = Config(
                append_raw_features=False,
                dataset=dataset,
                force_decision_tree=True,
                include_description=False,
                max_tree_depth=0,
                method=method,
                num_examples=1,
                num_iters=5,
                num_trees=5,
                root=os.getcwd(),
            )
            tree_files = []
            for iter in range(config.num_trees):
                config.iter = iter
                tree_files.append(get_tree_path(config))

            key_count = count_keys_in_file(X.keys(), tree_files)
            embeddings = tree_to_embedding(X, config)
            dim_embedding = len(embeddings[0])
            embedding_dimensions[method][dataset] = dim_embedding
            num_features[method][dataset] = sum(1 for value in key_count.values() if value != 0)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    embedding_dimensions.to_csv(os.path.join(result_path, "embedding_dimensions.csv"), index=True,
                                index_label="Dataset")
    embedding_dimensions.to_latex(os.path.join(result_path, "embedding_dimensions.tex"), escape=True)
    num_features.to_csv(os.path.join(result_path, "num_features.csv"), index=True, index_label="Dataset")
    num_features.to_latex(os.path.join(result_path, "num_features.tex"), escape=True)


def check_inputs(max_tree_depth_list, num_examples_list, result_path, temperature_list, include_description_list):
    if temperature_list is None:
        raise ValueError("Temperature cannot be empty")
    if max_tree_depth_list is None:
        raise ValueError("Max tree depth cannot be empty")
    if num_examples_list is None:
        raise ValueError("Number of examples cannot be empty")
    if include_description_list is None:
        raise ValueError("Include description cannot be empty")
    if not os.path.exists(os.path.dirname(result_path)):
        os.makedirs(os.path.dirname(result_path))
