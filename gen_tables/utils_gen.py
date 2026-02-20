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
    'xgb': 'XGBoost'
}

REGIME_MAP = {
    'nogen': 'fwd',
    'gen': 'gen'
}

DOMEN_MAP = {
    'business': ['hiring', 'marketing', 'telco', 'callcenter'],
    'people_society': ['fitness', 'tech_mental_health_survey', 'adult', 'extrovert'],
    'finance': ['audit', 'credit', 'fraud', 'bank_credit_scoring'],
    'healthcare': ['cancer', 'diabetes', 'transfusion', 'postpartum'],
    'education': ['student_famsup', 'tae', 'irish', 'reading'],
    'software_engineering': ['pc4', 'kc1', 'steel', 'machine'],
    'law': ['compas', 'vote', 'san_francisco_crimes', 'crimes_arrest'],
    'natural_science': ['bbbp', 'biodegr', 'seismic_bumps', 'stars'],
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


def filter_df(df, config):

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
    filtered_df = filtered_df.replace([-999, np.nan], '-')

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


def get_table_shots(df, config):
    domain = DOMEN_MAP[config['domain']]
    df_filt = df.loc[df.index.get_level_values('Dataset').isin(domain)]

    if config['serialization']:
        if config['serialization'][0] in df_filt.index.get_level_values('Serialization').unique():
            df_filt = df_filt.xs(config['serialization'][0], level='Serialization', drop_level=True)
        else:
            print(f"Warning: Serialization '{config['serialization']}' not found. Showing all.")

    cols_to_keep = [c for c in df_filt.columns if c[0] in config['shots']]
    df_filt = df_filt[cols_to_keep]

    cols_to_keep = [c for c in df_filt.columns if c[2] in config['models']]
    df_filt = df_filt[cols_to_keep]

    df_stacked = df_filt.stack(level='Shots')

    df_stacked = df_stacked.reorder_levels(['Model', 'Regime'], axis=1)
    df_stacked = df_stacked.sort_index(axis=1)

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


    num_data_cols = 2
    num_model_cols = len(df_stacked.columns) - 2
    col_spec = "ll" + "c" * num_model_cols
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")

    latex_lines.append(f"& & \\multicolumn{{{num_model_cols}}}{{c}}{{\\textbf{{Models}}}} \\\\")
    latex_lines.append(f"\\cmidrule(lr){{3-{2+num_model_cols}}}")

    unique_models = df_stacked.columns[2:].get_level_values(0).unique()

    header_row_2 = ["Dataset", "Shot"]
    cmidrule_ranges = []
    current_col_idx = 3

    for mod in unique_models:
        count = df_stacked.columns[2:].get_level_values(0).tolist().count(mod)
        header_row_2.append(f"\\multicolumn{{{count}}}{{c}}{{{mod}}}")
        cmidrule_ranges.append((current_col_idx, current_col_idx + count - 1))
        current_col_idx += count

    latex_lines.append(" & ".join(header_row_2) + " \\\\")


    for i, (start, end) in enumerate(cmidrule_ranges):
        latex_lines.append(f"\\cmidrule(lr){{{start}-{end}}}")

    sub_header_row = ["", ""]
    for mod in unique_models:
        sub_cols = df_stacked.columns[2:][df_stacked.columns[2:].get_level_values(0) == mod].get_level_values(1)
        for reg in sub_cols:
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
                ds_cell = f"\\multirow{{{num_rows}}}{{*}}{{{ds}}}"
            else:
                ds_cell = ""


            shot_cell = row[('Shots', '')]

            data_cells = []
            for col in df_stacked.columns[2:]:
                val = row[col]
                data_cells.append(format_value(val))

            line_parts = [ds_cell, shot_cell] + data_cells
            latex_lines.append(" & ".join(line_parts) + " \\\\")

        if i < len(unique_datasets) - 1:
            latex_lines.append("\\midrule")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("")
    latex_lines.append(f"\\caption{{{config['caption']}}}")
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

    df_filt = df_filt.reorder_levels(['Model', 'Shots', 'Regime'], axis=1)
    df_filt = df_filt.sort_index(axis=1)

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

    unique_models = df_stacked.columns[2:].get_level_values(0).unique()

    header_row = ["Dataset", "Serialization"]
    cmidrule_ranges = []
    current_col_idx = 3

    for mod in unique_models:
        count = df_stacked.columns[2:].get_level_values(0).tolist().count(mod)
        header_row.append(f"\\multicolumn{{{count}}}{{c}}{{{mod}}}")
        cmidrule_ranges.append((current_col_idx, current_col_idx + count - 1))
        current_col_idx += count

    latex_lines.append(" & ".join(header_row) + " \\\\")

    for start, end in cmidrule_ranges:
        latex_lines.append(f"\\cmidrule(lr){{{start}-{end}}}")

    sub_header = ["", ""]
    for mod in unique_models:
        sub_cols = df_stacked.columns[2:][
            df_stacked.columns[2:].get_level_values(0) == mod
        ].get_level_values(1)

        for shot in sub_cols:
            sub_header.append(shot)

    latex_lines.append(" & ".join(sub_header) + " \\\\")
    latex_lines.append("\\midrule")

    unique_datasets = df_stacked["Dataset"].unique()

    for i, ds in enumerate(unique_datasets):

        ds_rows = df_stacked[df_stacked["Dataset"] == ds]
        ds_rows = ds_rows.reindex(columns=sorted(ds_rows.columns, key=shot_sort_key))
        num_rows = len(ds_rows)

        for idx, row in ds_rows.iterrows():

            if idx == ds_rows.index[0]:
                ds_cell = f"\\multirow{{{num_rows}}}{{*}}{{{ds}}}"
            else:
                ds_cell = ""

            serialization_cell = str(row[('Serialization', '')]).replace('_', r'\_')

            data_cells = []
            for col in df_stacked.columns[2:]:
                data_cells.append(str(row[col]))

            latex_lines.append(
                " & ".join([ds_cell, serialization_cell] + data_cells) + " \\\\"
            )

        if i < len(unique_datasets) - 1:
            latex_lines.append("\\midrule")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("}")
    latex_lines.append("")
    latex_lines.append(f"\\caption{{{config['caption']}}}")
    latex_lines.append(f"\\label{{{config['label']}}}")
    latex_lines.append("\\end{table*}")

    latex_content = "\n".join(latex_lines)

    file_path = Path(config['tables_path']) / config['table_types'] / f"{config['domain']}.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)


    return latex_content