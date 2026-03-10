from pathlib import Path
import yaml
import pandas as pd
import numpy as np

from gen_tables.utils_gen import (
    combine_mean_std_columns, 
    filter_df,
    get_table_shots,
    get_table_serializations,
)

script_dir = Path(__file__).parent
with open(script_dir.parent / "config.yaml", "r") as f:
    config = yaml.safe_load(f)


# add automatic table formatting from raw results files

df = pd.read_csv(
    config['data']['TGEN_LOCAL_DATASET_PATH'],
    header=[0, 1, 2, 3],     
    index_col=[0, 1],
)

common_settings = {
    'tables_path': 'results/latex_tables_1/',
    'domain': 'healthcare',
    'shots': ['0', '4', '8', '16', '32', '64'],
    'models': ['qwen38b', 'qwen314b', 'gpt4omini', 'tabpfn', 'logreg', 'rf', 'gboost'],
    'metrics': ['roc_auc_mean', 'roc_auc_std'],
}

domains = ['business', 'people_society', 'finance', 'healthcare', 'education', 'software_engineering', 'law', 'natural_science', 'synthetic']
table_modes = ['shots', 'serializations']


def main():
    for domain in domains:
        for table_mode in table_modes: 

            if table_mode == 'shots':
                config = {
                    'table_types': 'shots',
                    'serialization': ['feat_val', 'feat_val_mask', 'html', 'markdown', 'markdown_mask'],
                    'regimes': ['gen', 'nogen'],
                    'caption': 'Healthcare - Shot Results',
                    'label': 'tab:shot_results',
                    'model_orders': [('qwen38b', 'gen'),
                                     ('qwen38b', 'nogen'),
                                     ('qwen314b', 'gen'),
                                     ('qwen314b', 'nogen'),
                                     ('gpt4omini', 'gen'),
                                     ('tabpfn', 'nogen'),
                                     ('logreg', 'nogen'),
                                     ('rf', 'nogen'),
                                     ('gboost', 'nogen')],
                    **common_settings  
                }
            elif table_mode == 'serializations':
                config = {
                    'table_types': 'serializations',
                    'serialization': ['feat_val', 'feat_val_mask', 'html', 'markdown', 'markdown_mask'],
                    'regimes': ['gen', 'nogen'],
                    'caption': 'Healthcare - Serialization Results',
                    'label': 'tab:serialization_results',
                    'model_orders': [('Dataset', ''),
                                     ('Serialization', ''),
                                     ('Qwen3-8B', 'shot 0'),
                                     ('Qwen3-8B', 'shot 4'),
                                     ('Qwen3-8B', 'shot 8'),
                                     ('Qwen3-8B', 'shot 16'),
                                     ('Qwen3-8B', 'shot 32'),
                                     ('Qwen3-8B', 'shot 64'),
                                     ('Qwen3-14B', 'shot 0'),
                                     ('Qwen3-14B', 'shot 4'),
                                     ('Qwen3-14B', 'shot 8'),
                                     ('Qwen3-14B', 'shot 16'),
                                     ('Qwen3-14B', 'shot 32'),
                                     ('Qwen3-14B', 'shot 64'),
                                     ('GPT-4o-mini', 'shot 0'),
                                     ('GPT-4o-mini', 'shot 4'),
                                     ('GPT-4o-mini', 'shot 8'),
                                     ('GPT-4o-mini', 'shot 16'),
                                     ('GPT-4o-mini', 'shot 32'),
                                     ('GPT-4o-mini', 'shot 64')],
                    **common_settings  
                }
            else:
                raise ValueError(f"Invalid table_mode: {table_mode}")

            config['domain'] = domain
            filtered_df = filter_df(df, config)
            df_combined = combine_mean_std_columns(filtered_df, decimal_places=3)

            if config['table_types'] == 'shots':
                config['caption'] = f'{config.get('domain')} - {config.get('table_types')}'
                latex_table = get_table_shots(df_combined, config)
            elif config['table_types'] == 'serializations':
                config['caption'] = f'{config.get('domain')} - {config.get('table_types')}'
                latex_table = get_table_serializations(df_combined, config)


if __name__ == "__main__":
    main()