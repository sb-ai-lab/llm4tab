from pathlib import Path
import yaml
import pandas as pd
import numpy as np

from gen_tables.utils_gen import (
    combine_mean_std_columns, 
    filter_df,
    get_table_shots,
    get_table_serializations,
    get_table_zero_shot_by_dataset,
    get_table_zero_shot_vs_tabpfn,
    get_table_context_configs,
    get_table_models_vs_shots,
    get_table_delta_context_vs_nocontext,
    get_table_synthetic_models_shots,
    get_table_icl_combinations,
    get_table_new_datasets_zero_shot,
    get_table_shots_vs_models_new_datasets,
    patch_external_metrics,
    reset_aggregated_md,
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
df = patch_external_metrics(df)

common_settings = {
    'tables_path': 'table_gen_results/latex_tables/',
    'domain': 'healthcare',
    'shots': ['0', '4', '8', '16', '32', '64'],
    'models': ['qwen38b', 'qwen314b', 'gpt4omini', 'tabpfn', 'logreg', 'rf', 'gboost'],
    'metrics': ['roc_auc_mean', 'roc_auc_std'],
}

domains = ['business', 'people_society', 'finance', 'healthcare', 'education', 'software_engineering', 'law', 'natural_science', 'synthetic']
table_modes = ['shots', 'serializations']


def main():
    reset_aggregated_md()
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
                config['caption'] = f"{config.get('domain')} - {config.get('table_types')}"
                latex_table = get_table_shots(df_combined, config)
            elif config['table_types'] == 'serializations':
                config['caption'] = f"{config.get('domain')} - {config.get('table_types')}"
                latex_table = get_table_serializations(df_combined, config)

            config_rq1 = {
                'table_types': 'zero_shot_vs_tabpfn',
                'domain': 'all',
                'serialization': ['feat_val'],
                'regimes': ['gen', 'nogen'],
                'shots': ['0', '16'],
                'models': ['gpt4omini', 'qwen317b', 'qwen38b', 'qwen314b', 'tabpfn'],
                'metrics': ['roc_auc_mean', 'roc_auc_std'],
                'tables_path': 'table_gen_results/latex_tables/',
                'caption': 'Zero-shot LLMs vs TabPFN (16-shot) across all domains',
                'label': 'tab:zero_shot_vs_tabpfn_all',
            }
            filtered_df = filter_df(df, config_rq1)
            latex_table = get_table_zero_shot_vs_tabpfn(filtered_df, config_rq1)

            config_rq2 = {
                'tables_path': 'table_gen_results/latex_tables/',
                'caption': 'Context configurations comparison across models (0-shot)',
                'label': 'tab:context_configs',
            }
            latex_table = get_table_context_configs(config_rq2)

            config_rq3 = {
                'tables_path': 'table_gen_results/latex_tables/',
                'caption': 'Models vs Shots on Classic Datasets (Cols-Context)',
                'label': 'tab:models_vs_shots_classic',
                'serialization': ['feat_val']
            }
            latex_table = get_table_models_vs_shots(config_rq3)

            config_rq4 = {
                'tables_path': 'table_gen_results/latex_tables/',
                'caption': 'Delta (Cols-Context - NoCols-NoContext) across shots on Classic Datasets',
                'label': 'tab:delta_context_vs_nocontext',
            }
            latex_table = get_table_delta_context_vs_nocontext(config_rq4)

            config_rq5 = {
                'tables_path': 'table_gen_results/latex_tables/',
                'caption': 'Comparison of ROC-AUC between prompting regimes with 3 In-Context Learning Entities',
                'label': 'tab:icl_combinations',
            }
            latex_table = get_table_icl_combinations(config_rq5)

            config_rq6 = {
                'tables_path': 'table_gen_results/latex_tables/',
                'caption': 'Model performance on synthetic datasets across shots',
                'label': 'tab:synthetic_models_shots',
            }
            latex_table = get_table_synthetic_models_shots(config_rq6)

            config_rq0 = {
                'tables_path': 'table_gen_results/latex_tables/',
                'caption': 'Zero-shot performance by dataset (Classic datasets)',
                'label': 'tab:zero_shot_by_dataset',
            }
            latex_table = get_table_zero_shot_by_dataset(config_rq0)

            config_baselines_0shot = {
                'tables_path': 'table_gen_results/latex_tables/',
                'caption': 'Zero-shot performance on new datasets',
                'label': 'tab:new_datasets_zero_shot',
            }
            latex_table = get_table_new_datasets_zero_shot(config_baselines_0shot)

            config_baselines_fewshot = {
                'tables_path': 'table_gen_results/latex_tables/',
                'caption': 'Model performance across shots on New datasets',
                'label': 'tab:shots_vs_models_new',
            }
            latex_table = get_table_shots_vs_models_new_datasets(config_baselines_fewshot)

            


if __name__ == "__main__":
    main()