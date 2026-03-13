import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path


MODEL_MAP = {
    'qwen314b': 'Qwen3-14B',
    'qwen38b': 'Qwen3-8B',
    'qwen31.7b': 'Qwen3-1.7B',
    'gpt4omini': 'GPT-4o-mini',
    'tabpfn': 'TabPFN',
    'logreg': 'LogReg',
    'rf': 'Random Forest',
    'gboost': 'XGBoost'
}

REGIME_MAP = {
    'nogen': 'fwd',
    'gen': 'gen'
}

DOMEN_MAP = {
    'business': ['hiring', 'marketing', 'telco', 'callcenter', 'hiring_ls_qwen', 'marketing_ls_qwen', 'telco_ls_qwen', 'callcenter_ls_qwen', 'hiring_ls_gpt', 'marketing_ls_gpt', 'telco_ls_gpt', 'callcenter_ls_gpt'],
    'people_society': ['fitness', 'tech_mental_health_survey', 'adult', 'extrovert', 'fitness_ls_qwen', 'tech_mental_health_survey_ls_qwen', 'adult_ls_qwen', 'extrovert_ls_qwen', 'fitness_ls_gpt', 'tech_mental_health_survey_ls_gpt', 'adult_ls_gpt', 'extrovert_ls_gpt'],
    'finance': ['audit', 'credit', 'fraud', 'bank_credit_scoring', 'audit_ls_qwen', 'credit_ls_qwen', 'fraud_ls_qwen', 'bank_credit_scoring_ls_qwen', 'audit_ls_gpt', 'credit_ls_gpt', 'fraud_ls_gpt', 'bank_credit_scoring_ls_gpt'],
    'healthcare': ['cancer', 'diabetes', 'transfusion', 'postpartum', 'cancer_ls_qwen', 'diabetes_ls_qwen', 'transfusion_ls_qwen', 'postpartum_ls_qwen', 'cancer_ls_gpt', 'diabetes_ls_gpt', 'transfusion_ls_gpt', 'postpartum_ls_gpt'],
    'education': ['student_famsup', 'tae', 'irish', 'reading', 'student_famsup_ls_qwen', 'tae_ls_qwen', 'irish_ls_qwen', 'reading_ls_qwen', 'student_famsup_ls_gpt', 'tae_ls_gpt', 'irish_ls_gpt', 'reading_ls_gpt'],
    'software_engineering': ['pc4', 'kc1', 'steel', 'machine', 'pc4_ls_qwen', 'kc1_ls_qwen', 'steel_ls_qwen', 'machine_ls_qwen', 'pc4_ls_gpt', 'kc1_ls_gpt', 'steel_ls_gpt', 'machine_ls_gpt'],
    'law': ['compas', 'vote', 'san_francisco_crimes', 'crimes_arrest', 'compas_ls_qwen', 'vote_ls_qwen', 'san_francisco_crimes_ls_qwen', 'crimes_arrest_ls_qwen', 'compas_ls_gpt', 'vote_ls_gpt', 'san_francisco_crimes_ls_gpt', 'crimes_arrest_ls_gpt'],
    'natural_science': ['bbbp', 'biodegr', 'seismic_bumps', 'stars_ls_qwen', 'bbbp_ls_qwen', 'biodegr_ls_qwen', 'seismic_bumps_ls_qwen', 'stars_ls_qwen', 'bbbp_ls_gpt', 'biodegr_ls_gpt', 'seismic_bumps_ls_gpt', 'stars_ls_gpt'],
    'synthetic': ['mlp0_f5_h0_no_noise_ReLU', 
                  'mlp1_f10_h10_no_noise_ReLU', 
                  'mlp2_f15_h15_no_noise_ReLU', 
                  'mlp3_f20_h20_no_noise_ReLU', 
                  'mlp5_f30_h30_no_noise_ReLU', 
                  'mlp7_f40_h40_no_noise_ReLU', 
                  'mlp9_f50_h50_no_noise_ReLU',   
                  ]
}

SERIALIZATION_MAP = {
    '1_old': 'feat_val',
    '3_old': 'feat_val_mask',
    'html_new': 'html',
    'markdown_new': 'markdown',
    'markdown_masked_new': 'markdown_mask',
}

def combine_mean_std_columns(df, decimal_places=3):


    col_tuples = df.columns.tolist()
    grouped = defaultdict(dict)

    for col in col_tuples:
        shot, regime, model, metric = col
        key = (shot, regime, model)
        grouped[key][metric] = col

    new_data = {}
    new_columns = []

    for key, metrics in grouped.items():
        shot, regime, model = key

        if 'roc_auc_mean' in metrics and 'roc_auc_std' in metrics:
            mean_col = metrics['roc_auc_mean']
            std_col = metrics['roc_auc_std']

            mean_vals = df[mean_col]
            std_vals = df[std_col]

            combined = []
            for m, s in zip(mean_vals, std_vals):
                if m == '-' or s == '-' or pd.isna(m) or pd.isna(s):
                    combined.append('-')
                else:
                    combined.append(f'{float(m):.{decimal_places}f} ± {float(s):.{decimal_places}f}')

            new_columns.append(key)
            new_data[key] = combined

        elif 'f1_mean' in metrics and 'f1_std' in metrics:
            mean_col = metrics['f1_mean']
            std_col = metrics['f1_std']

            mean_vals = df[mean_col]
            std_vals = df[std_col]

            combined = []
            for m, s in zip(mean_vals, std_vals):
                if m == '-' or s == '-' or pd.isna(m) or pd.isna(s):
                    combined.append('-')
                else:
                    combined.append(f'{float(m):.{decimal_places}f} ± {float(s):.{decimal_places}f}')

            new_columns.append(key)
            new_data[key] = combined
        else:
                break

    df_combined = pd.DataFrame(new_data, index=df.index)
    df_combined.columns = pd.MultiIndex.from_tuples(new_columns,
                                                     names=['Shots', 'Regime', 'Model'])


    return df_combined


def map_serializations(df, mapping=SERIALIZATION_MAP):

    if df.empty or 'Serialization' not in df.index.names:
        return df
    
    df = df.copy()
    
    idx_df = df.index.to_frame()
    
    idx_df['Serialization'] = idx_df['Serialization'].map(mapping).fillna(idx_df['Serialization'])
    
    df.index = pd.MultiIndex.from_frame(idx_df)
    
    return df

def filter_df(df, config):

    df = df.copy()

    df = map_serializations(df)

    domain = DOMEN_MAP[config['domain']]
    
    row_mask = (
        df.index.get_level_values('Dataset').isin(domain) &
        df.index.get_level_values('Serialization').isin(config['serialization'])
    )

    col_mask = (
        df.columns.get_level_values('Shots').isin(config['shots']) &
        df.columns.get_level_values('Regime').isin(config['regimes']) &
        df.columns.get_level_values('Model').isin(config['models']) &
        df.columns.get_level_values('Metric').isin(config['metrics'])
    )


    filtered_df = df.loc[row_mask, col_mask]
    
    if not filtered_df.empty:
        filtered_df = filtered_df.replace(-999, '-')
        filtered_df = filtered_df.fillna('-')


    return filtered_df


def format_value(val):
    if val is None or val == '-' or pd.isna(val):
        return '--'
    val_str = str(val).strip()
    if '±' in val_str:
        parts = val_str.split('±')
        mean = parts[0].strip()
        std = parts[1].strip()
        return f"\\num{{{mean} \\pm {std}}}"
    return f"\\num{{{val_str}}}"


def escape_latex(text):
    if text is None:
        return ""
    s = str(text)
    s = s.replace('_', '\\_')
    return s


def get_table_shots(df, config):
    domain = DOMEN_MAP[config['domain']]
    df_filt = df.loc[df.index.get_level_values('Dataset').isin(domain)]
    
    if config['domain'] == 'synthetic':
        df_filt = df_filt.loc[df.index.get_level_values('Serialization').isin(['feat_val_mask'])]
    else:
        df_filt = df_filt.loc[df.index.get_level_values('Serialization').isin(['feat_val'])]

    cols_to_keep = [c for c in df_filt.columns if c[0] in config['shots']]
    df_filt = df_filt[cols_to_keep]

    cols_to_keep = [c for c in df_filt.columns if c[2] in config['models']]
    df_stacked = df_filt.stack(level='Shots')

    df_stacked = df_stacked.reorder_levels(['Model', 'Regime'], axis=1)
    df_stacked = df_stacked.sort_index(axis=1)
    df_stacked = df_stacked[config['model_orders']]

    new_cols = []
    for mod, reg in df_stacked.columns:
        mod_name = MODEL_MAP.get(mod, mod)
        reg_name = REGIME_MAP.get(reg, reg)
        new_cols.append((mod_name, reg_name))
    df_stacked.columns = pd.MultiIndex.from_tuples(new_cols, names=['Models', 'Regime'])

    df_stacked = df_stacked.reset_index()
    latex_lines = []

    latex_lines.append("\\begin{table*}[ht]")
    latex_lines.append("\\sisetup{")
    latex_lines.append("  separate-uncertainty = true,")
    latex_lines.append("  table-align-uncertainty = true,")
    latex_lines.append("  table-figures-uncertainty = 1,")
    latex_lines.append("}")
    latex_lines.append("\\centering")
    latex_lines.append("\\scriptsize")
    latex_lines.append("\\setlength{\\tabcolsep}{2.5pt}")
    latex_lines.append("\\renewcommand{\\arraystretch}{1.05}")
    latex_lines.append("")

    model_columns = [col for col in df_stacked.columns[2:] if col[0] != 'Shots']
    num_data_cols = 2
    num_model_cols = len(model_columns)
    
    col_spec = "ll" + "c" * num_model_cols
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")

    latex_lines.append(f"& & \\multicolumn{{{num_model_cols}}}{{c}}{{\\textbf{{Models}}}} \\\\")
    latex_lines.append(f"\\cmidrule(lr){{3-{2+num_model_cols}}}")

    unique_models = []
    for col in model_columns:
        if col[0] not in unique_models:
            unique_models.append(col[0])

    header_row_2 = ["Dataset", "Shots"]
    cmidrule_ranges = []
    current_col_idx = 3

    for mod in unique_models:
        count = len([col for col in model_columns if col[0] == mod])
        header_row_2.append(f"\\multicolumn{{{count}}}{{c}}{{{mod}}}")
        cmidrule_ranges.append((current_col_idx, current_col_idx + count - 1))
        current_col_idx += count

    latex_lines.append(" & ".join(header_row_2) + " \\\\")

    for i, (start, end) in enumerate(cmidrule_ranges):
        latex_lines.append(f"\\cmidrule(lr){{{start}-{end}}}")

    sub_header_row = ["", ""]
    for mod in unique_models:
        sub_cols = [col[1] for col in model_columns if col[0] == mod]
        for reg in sub_cols:
            if mod in ['TabPFN', 'Random Forest', 'LogReg', 'XGBoost']:
                sub_header_row.append('/')
            else:
                sub_header_row.append(reg)

    latex_lines.append(" & ".join(sub_header_row) + " \\\\")
    latex_lines.append("\\midrule")

    unique_datasets = df_stacked['Dataset'].unique()

    for i, ds in enumerate(unique_datasets):
        ds_rows = df_stacked[df_stacked['Dataset'] == ds]
        ds_rows = ds_rows.sort_values(by='Shots', key=lambda col: pd.to_numeric(col))
        num_rows = len(ds_rows)

        for j, row in ds_rows.iterrows():
            if j == ds_rows.index[0]:
                ds_cell = f"\\multirow{{{num_rows}}}{{*}}{{{escape_latex(ds)}}}"
            else:
                ds_cell = ""

            shot_cell = row[('Shots', '')]

            data_cells = []

            for col in model_columns:
                val = row[col]
                data_cells.append(format_value(val))

            line_parts = [ds_cell, shot_cell] + data_cells
            latex_lines.append(" & ".join(line_parts) + " \\\\")

        if i < len(unique_datasets) - 1:
            latex_lines.append("\\midrule")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("")
    latex_lines.append(f"\\caption{{{escape_latex(config['caption'])}}}")
    latex_lines.append(f"\\label{{{config['label']}}}")
    latex_lines.append("\\end{table*}")

    latex_content = "\n".join(latex_lines)

    file_path = Path(config['tables_path']) / config['table_types'] / f"{config['domain']}.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)

    return latex_content


def shot_sort_key(col):
    if isinstance(col, tuple) and col[1].startswith("shot"):
        model, shot = col
        shot_number = int(shot.replace("shot ", ""))
        return (model, shot_number)
    else:
        return ("", -1)
    
def get_table_serializations(df, config):

    domain = DOMEN_MAP[config['domain']]
    df_filt = df.loc[df.index.get_level_values('Dataset').isin(domain)]

    cols_to_keep = [c for c in df_filt.columns if c[0] in config['shots']]
    df_filt = df_filt[cols_to_keep]

    cols_to_keep = [c for c in df_filt.columns if c[2] in config['models']]
    df_filt = df_filt[cols_to_keep]

    mask = df_filt.columns.get_level_values('Regime') == 'gen'
    df_filt = df_filt.loc[:, mask]

    original_columns = df_filt.columns.tolist()
    
    df_filt.columns = pd.MultiIndex.from_tuples(
        [(c[2], c[0], c[1]) for c in original_columns],  
        names=['Model', 'Shots', 'Regime']
    )

    if 'Regime' in df_filt.columns.names:
        df_filt.columns = df_filt.columns.droplevel('Regime')

    new_cols = []
    for model, shot in df_filt.columns:
        model_name = MODEL_MAP.get(model, model)
        new_cols.append((model_name, f"shot {shot}"))

    df_filt.columns = pd.MultiIndex.from_tuples(
        new_cols, names=["Models", "Shots"]
    )

    df_stacked = df_filt.reset_index()
    
    expected_columns = config['model_orders'][2:]

    current_columns = df_stacked.columns.tolist()
    
    col_position_map = {}
    for i, col in enumerate(expected_columns):
        col_position_map[col] = i
    
    fixed_columns = [('Dataset', ''), ('Serialization', '')]
    data_columns = [col for col in current_columns if col not in fixed_columns]
    
    expected_dict = {col: i for i, col in enumerate(expected_columns)}
    
    sorted_data_columns = []
    for col in data_columns:
        if col in expected_dict:
            sorted_data_columns.append((expected_dict[col], col))
    
    sorted_data_columns = [col for _, col in sorted(sorted_data_columns, key=lambda x: x[0])]
    
    for col in data_columns:
        if col not in expected_dict and col not in sorted_data_columns:
            sorted_data_columns.append(col)
    
    df_stacked = df_stacked[fixed_columns + sorted_data_columns]

    latex_lines = []

    latex_lines.append("\\begin{table*}[ht]")
    latex_lines.append("\\sisetup{")
    latex_lines.append("  separate-uncertainty = true,")
    latex_lines.append("  table-align-uncertainty = true,")
    latex_lines.append("  table-figures-uncertainty = 1,")
    latex_lines.append("}")
    latex_lines.append("\\centering")
    latex_lines.append("\\scriptsize")
    latex_lines.append("\\setlength{\\tabcolsep}{3pt}")
    latex_lines.append("\\renewcommand{\\arraystretch}{1.05}")
    latex_lines.append("")
    latex_lines.append("\\adjustbox{max width=\\textwidth}{")

    num_model_cols = len(df_stacked.columns) - 2
    col_spec = "ll" + "c" * num_model_cols

    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")

    latex_lines.append(
        f"& & \\multicolumn{{{num_model_cols}}}{{c}}{{\\textbf{{Models}}}} \\\\"
    )
    latex_lines.append(f"\\cmidrule(lr){{3-{2+num_model_cols}}}")

    unique_models = []
    for col in sorted_data_columns:
        model = col[0]
        if model not in unique_models:
            unique_models.append(model)

    header_row = ["Dataset", "Serialization"]
    cmidrule_ranges = []
    current_col_idx = 3

    for mod in unique_models:
        count = sum(1 for col in sorted_data_columns if col[0] == mod)
        header_row.append(f"\\multicolumn{{{count}}}{{c}}{{{mod}}}")
        cmidrule_ranges.append((current_col_idx, current_col_idx + count - 1))
        current_col_idx += count

    latex_lines.append(" & ".join(header_row) + " \\\\")

    for start, end in cmidrule_ranges:
        latex_lines.append(f"\\cmidrule(lr){{{start}-{end}}}")

    sub_header = ["", ""]
    for mod in unique_models:
        for col in sorted_data_columns:
            if col[0] == mod:
                sub_header.append(col[1])

    latex_lines.append(" & ".join(sub_header) + " \\\\")
    latex_lines.append("\\midrule")
    
    unique_datasets = df_stacked["Dataset"].unique()

    for i, ds in enumerate(unique_datasets):
        ds_rows = df_stacked[df_stacked["Dataset"] == ds]
        
        ds_rows = ds_rows.sort_values(by=('Serialization', ''))
        num_rows = len(ds_rows)

        for idx, row in ds_rows.iterrows():
            if idx == ds_rows.index[0]:
                ds_cell = f"\\multirow{{{num_rows}}}{{*}}{{{escape_latex(ds)}}}"
            else:
                ds_cell = ""

            serialization_cell = str(row[('Serialization', '')]).replace('_', r'\_')

            data_cells = []
            for col in sorted_data_columns:
                val = row[col]
                if val == '-':
                    data_cells.append('--')
                else:
                    data_cells.append(format_value(val))

            latex_lines.append(
                " & ".join([ds_cell, serialization_cell] + data_cells) + " \\\\"
            )

        if i < len(unique_datasets) - 1:
            latex_lines.append("\\midrule")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("}")
    latex_lines.append("")
    latex_lines.append(f"\\caption{{{escape_latex(config['caption'])}}}")
    latex_lines.append(f"\\label{{{config['label']}}}")
    latex_lines.append("\\end{table*}")

    latex_content = "\n".join(latex_lines)

    file_path = Path(config['tables_path']) / config['table_types'] / f"{config['domain']}.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)

    return latex_content