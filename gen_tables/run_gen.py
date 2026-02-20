import pandas as pd
import numpy as np

from gen_tables.utils_gen import (
    combine_mean_std_columns, 
    filter_df,
    get_table_shots,
    get_table_serializations,
)


df = pd.read_csv(
    "datasets/agr_real_prompt_1.csv",
    header=[0, 1, 2, 3],     
    index_col=[0, 1],       
)


common_settings = {
    'tables_path': 'results/latex_tables/',
    'domain': 'healthcare',
    'shots': ['0', '16', '64'],
    'models': ['qwen38b', 'qwen314b', 'gpt4omini', 'tabpfn'],
    'metrics': ['roc_auc_mean', 'roc_auc_std'],
}

domains = ['business', 'people_society', 'finance', 'healthcare', 'education', 'software_engineering', 'law', 'natural_science']
table_modes = ['shots', 'serializations']


def main():
    for domain in domains:
        for table_mode in table_modes: 

            if table_mode == 'shots':
                config = {
                    'table_types': 'shots',
                    'serialization': ['1_old'],
                    'regimes': ['gen', 'nogen'],
                    'caption': 'Healthcare - Shot Results',
                    'label': 'tab:shot_results',
                    **common_settings  
                }
            elif table_mode == 'serializations':
                config = {
                    'table_types': 'serializations',
                    'serialization': ['1_old', '3_old', 'html_new', 'markdown_masked', 'markdown_masked_new'],
                    'regimes': ['gen'],
                    'caption': 'Healthcare - Serialization Results',
                    'label': 'tab:shot_results',
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