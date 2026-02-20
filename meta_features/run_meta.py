
import os
import yaml
from transformers import AutoTokenizer
import re
from pathlib import Path
import numpy as np
import pandas as pd
import json
from tqdm import tqdm

from source.utils import (
    get_df, transform_dataset,
)
from source.prompts import (
    prompt_by_df, system_prompt
)
from source.models import (
    process_serialization
)

from meta_features.utils_meta import (
    DatasetMetaFeatures, get_prompt_metadata, compute_metrics_for_dataset, openai_chat_template
)

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


MODEL_NAME = "Qwen/Qwen3-1.7B" #"openai/gpt-4o-mini" #"Qwen/Qwen3-1.7B"
if MODEL_NAME == "Qwen/Qwen3-1.7B":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def main():
    config = load_config()
    exp_cfg = config['experiment']
    schema = json.loads(exp_cfg['SCHEMA'])
    model_family = MODEL_NAME.lower().split('/')[0]
    all_shots = exp_cfg['shot_list']

    data_origin = (
        config['data']['DF_TYPE'] + '_' + config['openml']['type']
        if config['data']['DF_TYPE'] == 'openml'
        else config['data']['DF_TYPE']
    )

    if config['data']['DF_TYPE'] == 'custom':
        dataset_files = [f.lower().split('.')[0] for f in os.listdir(config['data']['LOCAL_DATASET_PATH'])]
    elif config['data']['DF_TYPE'] == 'openml':
        dataset_files = config['openml']['df_openml']

    for shot in all_shots:
        exp_cfg['N_SHOTS'] = shot
        results_data = []
        datasets_features = []
        for dataset_name in tqdm(dataset_files, desc="Datasets"):
            config['data']['DATASET_NAME'] = dataset_name
            X_train, y_train, X_test, y_test = get_df(config)
            df_train = pd.concat([transform_dataset(X_train, config), y_train], axis=1).dropna()
            df_test = pd.concat([transform_dataset(X_test, config), y_test], axis=1).dropna()
            task_description = prompt_by_df.get(dataset_name, None)
            for serialization in exp_cfg['serialization_list']:
                serialization_train, serialization_test = process_serialization(
                    serialization, df_train, df_test,
                    exp_cfg['N_SHOTS'], exp_cfg['RATIO'], exp_cfg['SAMPLING_REGIME'], exp_cfg['RANDOM_STATE']
                )

                few_shot_examples = f"{serialization_train}" if exp_cfg['N_SHOTS'] > 0 else ""
                
                if MODEL_NAME == "Qwen/Qwen3-1.7B":
                    avg_metrics = compute_metrics_for_dataset(
                        serialization_test, config, schema, system_prompt,
                        task_description, few_shot_examples, model_family, 
                        MODEL_NAME, tokenizer
                )
                else:
                    avg_metrics = compute_metrics_for_dataset(
                        serialization_test, config, schema, system_prompt,
                        task_description, few_shot_examples, model_family, 
                        MODEL_NAME
                    )

                row = {
                    "model": model_family,
                    "dataset": dataset_name,
                    "serialization": serialization,
                    **avg_metrics
                }
                results_data.append(row)

            calculator = DatasetMetaFeatures(X_train, dataset_name=dataset_name)
            features = calculator.calculate_all()
            datasets_features.append(features)

        prompt_feat = pd.DataFrame(results_data)
        dataset_feat = pd.DataFrame(datasets_features)
        results_data = pd.merge(prompt_feat, dataset_feat, on='dataset')


        file_path = Path('results/new_features/') / model_family / str(exp_cfg['N_SHOTS']) / f"{data_origin}.csv"
        print(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results_data).to_csv(file_path, index=False)


if __name__ == "__main__":
    main()