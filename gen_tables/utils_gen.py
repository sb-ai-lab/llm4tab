import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
import re


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

CLASSIC_BASE = [
    'hiring', 'adult', 'seismic_bumps', 'san_francisco_crimes',
    'tech_mental_health_survey', 'bbbp', 'audit', 'fraud',
    'telco', 'pc4', 'tae', 'irish', 'compas', 'vote',
    'cancer', 'steel', 'kc1', 'credit', 'transfusion',
    'fitness', 'diabetes', 'biodegr', 'marketing', 'student_famsup'
]

NEW_BASE = [
    'stars', 'machine', 'callcenter', 'bank_credit_scoring',
    'extrovert', 'reading', 'postpartum', 'crimes_arrest'
]

LS_GPT = [
    'hiring_ls_gpt', 'adult_ls_gpt', 'seismic_bumps_ls_gpt', 'san_francisco_crimes_ls_gpt',
    'tech_mental_health_survey_ls_gpt', 'bbbp_ls_gpt', 'audit_ls_gpt', 'fraud_ls_gpt',
    'telco_ls_gpt', 'pc4_ls_gpt', 'tae_ls_gpt', 'irish_ls_gpt', 'compas_ls_gpt', 'vote_ls_gpt',
    'cancer_ls_gpt', 'steel_ls_gpt', 'kc1_ls_gpt', 'credit_ls_gpt', 'transfusion_ls_gpt',
    'fitness_ls_gpt', 'diabetes_ls_gpt', 'biodegr_ls_gpt', 'marketing_ls_gpt', 'student_famsup_ls_gpt'
]

LS_QWEN = [
    'hiring_ls_qwen', 'adult_ls_qwen', 'seismic_bumps_ls_qwen', 'san_francisco_crimes_ls_qwen',
    'tech_mental_health_survey_ls_qwen', 'bbbp_ls_qwen', 'audit_ls_qwen', 'fraud_ls_qwen',
    'telco_ls_qwen', 'pc4_ls_qwen', 'tae_ls_qwen', 'irish_ls_qwen', 'compas_ls_qwen', 'vote_ls_qwen',
    'cancer_ls_qwen', 'steel_ls_qwen', 'kc1_ls_qwen', 'credit_ls_qwen', 'transfusion_ls_qwen',
    'fitness_ls_qwen', 'diabetes_ls_qwen', 'biodegr_ls_qwen', 'marketing_ls_qwen', 'student_famsup_ls_qwen'
]


DOMEN_MAP['classic'] = CLASSIC_BASE
DOMEN_MAP['new'] = NEW_BASE
DOMEN_MAP['all'] = DOMEN_MAP['classic'] + DOMEN_MAP['new']
DOMEN_MAP['ls_gpt'] = LS_GPT
DOMEN_MAP['ls_qwen'] = LS_QWEN

SERIALIZATION_MAP = {
    '1_old': 'feat_val',
    '3_old': 'feat_val_mask',
    'html_new': 'html',
    'markdown_new': 'markdown',
    'markdown_masked_new': 'markdown_mask',
}

# =====================================================================
#  ВНЕШНИЕ ИСТОЧНИКИ МЕТРИК (новые файлы): TabPFN, TabICL, классические бейзлайны.
#  Метрики ЭТИХ моделей берутся не из основного df, а из отдельных файлов в
#  каталоге datasets/ (старый источник для них не используется). Пути относительные:
#  <корень проекта>/datasets/<файл>.  (= .../llm4tab/datasets/...)
# =====================================================================
import json as _json

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATASETS_DIR = _PROJECT_ROOT / "datasets"
_BENCHMARK_DIR = _PROJECT_ROOT / "BENCHMARK_RESULTS"

# Имена файлов — как в предыдущих ноутбуках. При необходимости правьте здесь.
EXTERNAL_FILES = {
    "baselines": {                       # классические ML-бейзлайны, long-format CSV
        "logreg": "lr_res__2_.csv",
        "rf":     "rf_res__2_.csv",
        "gboost": "gboost_res__2_.csv",
        "knn":    "knn_res__2_.csv",
        "naive":  "naive_res__2_.csv",
    },
    "tabpfn": [                          # wide-format (index=dataset, колонки=shots)
        "real_dfs_tabpfn3_seed5_FINAL_inv.csv",
        "mlp_dfs_tabpfn3_seed5_FINAL_inv.csv",
    ],
    "tabicl": "tabicl_5seeds_gpt.json",  # json
}
# Модели, метрики которых переопределяются из внешних файлов.
EXTERNAL_MODELS = set(EXTERNAL_FILES["baselines"].keys()) | {"tabpfn", "tabicl"}


def _ext_path(name):
    return _DATASETS_DIR / name


def _safe_float(x):
    try:
        if x is None:
            return None
        f = float(x)
        return None if (isinstance(f, float) and np.isnan(f)) else f
    except (TypeError, ValueError):
        return None


def _load_baselines_external():
    """{model_key: {dataset: {shot_str: {'roc_auc_mean':v, 'roc_auc_std':v}}}}"""
    out = {}
    for mkey, fname in EXTERNAL_FILES["baselines"].items():
        p = _ext_path(fname)
        if not p.exists():
            continue
        bdf = pd.read_csv(p)
        per_ds = {}
        for _, r in bdf.iterrows():
            ds = str(r["dataset"])
            shot = str(int(r["shots"]))
            per_ds.setdefault(ds, {})[shot] = {
                "roc_auc_mean": _safe_float(r.get("roc_auc_mean")),
                "roc_auc_std":  _safe_float(r.get("roc_auc_std")),
            }
        out[mkey] = per_ds
    return out


def _load_tabpfn_external():
    """{dataset: {shot_str: {'roc_auc_mean':v, 'roc_auc_std':v|None}}}
    Wide-формат: index=dataset, числовые shot-колонки = mean; '<shot>_std' = std (если есть)."""
    out = {}
    for fname in EXTERNAL_FILES["tabpfn"]:
        p = _ext_path(fname)
        if not p.exists():
            continue
        tdf = pd.read_csv(p)
        if "dataset" in tdf.columns:
            tdf = tdf.set_index("dataset")
        for ds, row in tdf.iterrows():
            d = out.setdefault(str(ds), {})
            for col in tdf.columns:
                cs = str(col)
                if cs.endswith("_std") and cs[:-4].isdigit():
                    shot = cs[:-4]
                    d.setdefault(shot, {})
                    d[shot]["roc_auc_std"] = _safe_float(row[col])
                elif cs.isdigit():
                    d.setdefault(cs, {})
                    d[cs]["roc_auc_mean"] = _safe_float(row[col])
                    d[cs].setdefault("roc_auc_std", None)
    return out


def _load_tabicl_external():
    """{dataset: {shot_str: {'roc_auc_mean':v, 'roc_auc_std':v|None}}}
    Парсер устойчив к нескольким вариантам схемы json:
      {ds: {shot: value}}  |  {ds: {shot: {mean, std}}}  |  {'tabicl': {...}}."""
    p = _ext_path(EXTERNAL_FILES["tabicl"])
    if not p.exists():
        return {}
    with open(p) as f:
        raw = _json.load(f)
    if isinstance(raw, dict):
        for k in list(raw.keys()):
            if str(k).lower() == "tabicl":
                raw = raw[k]
                break
    out = {}
    if isinstance(raw, dict):
        for ds, shots in raw.items():
            if not isinstance(shots, dict):
                continue
            d = out.setdefault(str(ds), {})
            for shot, val in shots.items():
                if isinstance(val, dict):
                    mean = val.get("roc_auc_mean", val.get("mean", val.get("roc_auc")))
                    std = val.get("roc_auc_std", val.get("std"))
                else:
                    mean, std = val, None
                d[str(shot)] = {"roc_auc_mean": _safe_float(mean),
                                "roc_auc_std": _safe_float(std)}
    return out


_EXTERNAL_CACHE = {}


def _get_external():
    if not _EXTERNAL_CACHE:
        _EXTERNAL_CACHE["baselines"] = _load_baselines_external()
        _EXTERNAL_CACHE["tabpfn"] = _load_tabpfn_external()
        _EXTERNAL_CACHE["tabicl"] = _load_tabicl_external()
    return _EXTERNAL_CACHE


def _external_value(model, dataset, shot, metric):
    ext = _get_external()
    if model in ext["baselines"]:
        src = ext["baselines"][model].get(str(dataset), {})
    elif model == "tabpfn":
        src = ext["tabpfn"].get(str(dataset), {})
    elif model == "tabicl":
        src = ext["tabicl"].get(str(dataset), {})
    else:
        return ("missing", None)
    cell = src.get(str(shot))
    if not cell or metric not in cell:
        return ("missing", None)
    return ("ok", cell[metric])


def patch_external_metrics(df):
    """Переопределяет в мультииндексном df (Shots, Regime, Model, Metric) метрики
    моделей из EXTERNAL_MODELS значениями из внешних файлов. Меняются только те
    ячейки, для которых нашлось значение во внешнем источнике; остальное (в т.ч.
    std, если во внешнем файле его нет) остаётся как есть. Не падает при
    отсутствии файлов/датасетов — тогда значения не меняются."""
    if df is None or getattr(df, "empty", True):
        return df
    if df.columns.nlevels != 4:
        return df
    df = df.copy()
    datasets = list(df.index.get_level_values(0))
    for col in list(df.columns):
        shot, regime, model, metric = col
        if model not in EXTERNAL_MODELS or metric not in ("roc_auc_mean", "roc_auc_std"):
            continue
        cur = list(df[col])
        new_col = []
        for ds, old in zip(datasets, cur):
            status, val = _external_value(model, ds, shot, metric)
            new_col.append(val if (status == "ok" and val is not None) else old)
        df[col] = new_col
    return df


# ---------------------------------------------------------------------
#  Сохранение таблиц: latex (.txt, как раньше) + .csv рядом + .md в
#  BENCHMARK_RESULTS (отдельный файл на таблицу + общий AGGREGATED_TABLE.md).
# ---------------------------------------------------------------------
def _clean_latex_cell(c):
    c = c.strip()
    c = re.sub(r"\\multirow\{[^}]*\}\{[^}]*\}\{([^}]*)\}", r"\1", c)
    c = re.sub(r"\\multicolumn\{[^}]*\}\{[^}]*\}\{([^}]*)\}", r"\1", c)
    c = re.sub(r"\\textbf\{([^}]*)\}", r"\1", c)
    c = re.sub(r"\\num\{([^}]*)\}", r"\1", c)
    c = c.replace("\\pm", "±").replace("\\_", "_").replace("\\,", " ")
    c = re.sub(r"\\[a-zA-Z]+", "", c)
    c = c.replace("{", "").replace("}", "")
    return c.strip()


def _latex_table_to_rows(latex_content):
    rows, in_tab = [], False
    for line in latex_content.splitlines():
        s = line.strip()
        if s.startswith("\\begin{tabular}"):
            in_tab = True
            continue
        if s.startswith("\\end{tabular}"):
            in_tab = False
            continue
        if not in_tab:
            continue
        if s.startswith("\\") and "&" not in s:      # rules / cmidrule
            continue
        if "&" not in s and not s.endswith("\\\\"):
            continue
        cl = s[:-2] if s.endswith("\\\\") else s
        cells = [_clean_latex_cell(c) for c in cl.split("&")]
        if any(c != "" for c in cells):
            rows.append(cells)
    return rows


def _rows_to_md(rows):
    if not rows:
        return ""
    w = max(len(r) for r in rows)
    rows = [r + [""] * (w - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |",
           "| " + " | ".join(["---"] * w) + " |"]
    for r in rows[1:]:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _md_name_for(file_path, config):
    if config and config.get("md_name"):
        return config["md_name"]
    sub = Path(file_path).parent.name
    m = re.match(r"(RQ\d+)", sub)
    return m.group(1) if m else sub


def reset_aggregated_md():
    """Очистить общий AGGREGATED_TABLE.md в начале прогона."""
    _BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    with open(_BENCHMARK_DIR / "AGGREGATED_TABLE.md", "w", encoding="utf-8") as f:
        f.write("# AGGREGATED_TABLE\n\n")


def save_all_formats(latex_content, file_path, config=None):
    """Сохранить таблицу в 3 форматах: latex .txt (как раньше), .csv рядом,
    .md в BENCHMARK_RESULTS (+ дописать в общий AGGREGATED_TABLE.md)."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:           # 1) latex
        f.write(latex_content)

    rows = _latex_table_to_rows(latex_content)
    if rows:                                                    # 2) csv рядом
        w = max(len(r) for r in rows)
        rows_p = [r + [""] * (w - len(r)) for r in rows]
        pd.DataFrame(rows_p).to_csv(file_path.with_suffix(".csv"),
                                    index=False, header=False, encoding="utf-8")

    _BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)           # 3) md
    md_table = _rows_to_md(rows)
    title = (config or {}).get("caption", file_path.stem)
    block = f"## {title}\n\n{md_table}\n"
    md_name = _md_name_for(file_path, config)
    with open(_BENCHMARK_DIR / f"{md_name}.md", "w", encoding="utf-8") as f:
        f.write(f"# {md_name}\n\n{block}")
    with open(_BENCHMARK_DIR / "AGGREGATED_TABLE.md", "a", encoding="utf-8") as f:
        f.write(block + "\n")
    return latex_content

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

    save_all_formats(latex_content, file_path, config)

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

    df_filt = df_filt.loc[:, df_filt.columns.get_level_values('Model') != 'tabpfn']

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

    save_all_formats(latex_content, file_path, config)

    return latex_content


def get_table_zero_shot_vs_tabpfn(df, config):

    new_datasets = [
        'stars', 'machine', 'callcenter', 'bank_credit_scoring',
        'extrovert', 'reading', 'postpartum', 'crimes_arrest'
    ]
    
    classic_datasets = [
        'hiring', 'adult', 'seismic_bumps', 'san_francisco_crimes',
        'tech_mental_health_survey', 'bbbp', 'audit', 'fraud',
        'telco', 'pc4', 'tae', 'irish', 'compas', 'vote',
        'cancer', 'steel', 'kc1', 'credit', 'transfusion',
        'fitness', 'diabetes', 'biodegr', 'marketing', 'student_famsup'
    ]
    
    all_datasets = classic_datasets + new_datasets
    
    col_specs = [
        ('gpt4omini', '0', 'gen'),
        ('qwen317b', '0', 'gen'),
        ('qwen38b', '0', 'gen'),
        ('qwen314b', '0', 'gen'),
        ('tabpfn', '16', 'nogen'),
    ]
    
    domain_datasets = DOMEN_MAP[config['domain']]
    df_filt = df.loc[df.index.get_level_values('Dataset').isin(domain_datasets)]
    
    if config['domain'] == 'synthetic':
        df_filt = df_filt.loc[df.index.get_level_values('Serialization').isin(['feat_val_mask'])]
    else:
        df_filt = df_filt.loc[df.index.get_level_values('Serialization').isin(['feat_val'])]
    
    results = []
    
    for group_name, group_datasets in [('All', all_datasets), ('Classic', classic_datasets), ('New', new_datasets)]:
        available_datasets = [ds for ds in group_datasets if ds in domain_datasets]
        
        row_data = {'Group': group_name}
        
        for model, shot, regime in col_specs:
            try:
                mean_col = df_filt.loc[:, (shot, regime, model, 'roc_auc_mean')]
                std_col = df_filt.loc[:, (shot, regime, model, 'roc_auc_std')]
                
                mean_vals = []
                std_vals = []
                
                for ds in available_datasets:
                    if ds in mean_col.index.get_level_values('Dataset'):
                        m = mean_col.loc[ds]
                        s = std_col.loc[ds]
                        if isinstance(m, pd.Series):
                            m = m.iloc[0]
                            s = s.iloc[0]
                        if m != '-' and not pd.isna(m):
                            mean_vals.append(float(m))
                            std_vals.append(float(s))
                
                if mean_vals:

                    agg_mean = np.mean(mean_vals)
                    agg_std = np.sqrt(np.mean(np.array(std_vals)**2))
                    row_data[f'{model}_{shot}_{regime}'] = (agg_mean, agg_std)
                else:
                    row_data[f'{model}_{shot}_{regime}'] = None
                    
            except KeyError:
                row_data[f'{model}_{shot}_{regime}'] = None
        
        results.append(row_data)
    

    latex_lines = []
    latex_lines.append("\\begin{table}[ht]")
    latex_lines.append("\\sisetup{")
    latex_lines.append("  separate-uncertainty = true,")
    latex_lines.append("  table-align-uncertainty = true,")
    latex_lines.append("  table-figures-uncertainty = 1,")
    latex_lines.append("}")
    latex_lines.append("\\centering")
    latex_lines.append("\\small")
    latex_lines.append("\\setlength{\\tabcolsep}{8pt}")
    latex_lines.append("")
    
    col_spec = "l" + "c" * len(col_specs)
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")
    
  
    header = ["Dataset Group"] + [
        "GPT-4o-mini\\\\0-shot", "Qwen3-1.7B\\\\0-shot", "Qwen3-8B\\\\0-shot", 
        "Qwen3-14B\\\\0-shot", "TabPFN\\\\16-shot"
    ]
    latex_lines.append(" & ".join(header) + " \\\\")
    latex_lines.append("\\midrule")
    
    for row in results:
        cells = [row['Group']]
        for model, shot, regime in col_specs:
            val = row.get(f'{model}_{shot}_{regime}')
            if val is not None:
                cells.append(f"\\num{{{val[0]:.3f} \\pm {val[1]:.3f}}}")
            else:
                cells.append("--")
        latex_lines.append(" & ".join(cells) + " \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append(f"\\caption{{{escape_latex(config['caption'])}}}")
    latex_lines.append(f"\\label{{{config['label']}}}")
    latex_lines.append("\\end{table}")
    
    latex_content = "\n".join(latex_lines)
    
    file_path = Path(config['tables_path']) / 'zero_shot_vs_tabpfn' / f"{config['domain']}.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_all_formats(latex_content, file_path, config)
    
    return latex_content

def get_table_zero_shot_vs_tabpfn(df, config):

    classic_datasets = CLASSIC_BASE
    new_datasets = NEW_BASE
    all_datasets = classic_datasets + new_datasets
    
    available_in_df = df.index.get_level_values('Dataset').unique().tolist()
    
    def get_base_name(full_name):
        if full_name.endswith('_ls_qwen'):
            return full_name[:-8]
        elif full_name.endswith('_ls_gpt'):
            return full_name[:-7]
        return full_name
    
    col_specs = [
        ('gpt4omini', '0', 'gen'),
        ('qwen317b', '0', 'gen'),
        ('qwen38b', '0', 'gen'),
        ('qwen314b', '0', 'gen'),
        ('tabpfn', '16', 'nogen'),
    ]
    
    results = []
    
    for group_name, group_datasets in [('All', all_datasets), ('Classic', classic_datasets), ('New', new_datasets)]:
        row_data = {'Group': group_name}
        
        for model, shot, regime in col_specs:
            mean_vals = []
            std_vals = []
            
            for ds_full in available_in_df:
                base_name = get_base_name(ds_full)
                if base_name not in group_datasets:
                    continue
                
                try:
                    m = df.loc[(ds_full, 'feat_val'), (shot, regime, model, 'roc_auc_mean')]
                    s = df.loc[(ds_full, 'feat_val'), (shot, regime, model, 'roc_auc_std')]
                    
                    if isinstance(m, pd.Series):
                        m = m.iloc[0]
                        s = s.iloc[0]
                    
                    if m != '-' and not pd.isna(m) and m != -999.0:
                        mean_vals.append(float(m))
                        std_vals.append(float(s))
                except (KeyError, ValueError, TypeError):
                    continue
            
            if mean_vals:
                agg_mean = np.mean(mean_vals)
                agg_std = np.sqrt(np.mean(np.array(std_vals)**2))
                row_data[f'{model}_{shot}_{regime}'] = (agg_mean, agg_std)
            else:
                row_data[f'{model}_{shot}_{regime}'] = None
        
        results.append(row_data)
    
    latex_lines = []
    latex_lines.append("\\begin{table}[ht]")
    latex_lines.append("\\sisetup{")
    latex_lines.append("  separate-uncertainty = true,")
    latex_lines.append("  table-align-uncertainty = true,")
    latex_lines.append("  table-figures-uncertainty = 1,")
    latex_lines.append("}")
    latex_lines.append("\\centering")
    latex_lines.append("\\small")
    latex_lines.append("\\setlength{\\tabcolsep}{8pt}")
    latex_lines.append("")
    
    col_spec = "l" + "c" * len(col_specs)
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")
    
    header = ["Dataset Group"] + [
        "GPT-4o-mini (gen)", "Qwen3-1.7B (gen)", "Qwen3-8B (gen)", 
        "Qwen3-14B (gen)", "TabPFN (16-shot)"
    ]
    latex_lines.append(" & ".join(header) + " \\\\")
    latex_lines.append("\\midrule")
    
    for row in results:
        cells = [row['Group']]
        for model, shot, regime in col_specs:
            val = row.get(f'{model}_{shot}_{regime}')
            if val is not None:
                cells.append(f"\\num{{{val[0]:.3f} \\pm {val[1]:.3f}}}")
            else:
                cells.append("--")
        latex_lines.append(" & ".join(cells) + " \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append(f"\\caption{{{escape_latex(config['caption'])}}}")
    latex_lines.append(f"\\label{{{config['label']}}}")
    latex_lines.append("\\end{table}")
    
    latex_content = "\n".join(latex_lines)
    
    file_path = Path(config['tables_path']) / 'RQ1_zero_shot_vs_tabpfn' / "all_domains.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_all_formats(latex_content, file_path, config)
    
    return latex_content

def get_table_context_configs(config):

    script_dir = Path(__file__).parent
    prompt1_path = script_dir.parent / "datasets" / "agr_all_prompt1.csv"
    prompt2_path = script_dir.parent / "datasets" / "agr_all_prompt2.csv"
    
    df_prompt1 = pd.read_csv(prompt1_path, header=[0, 1, 2, 3], index_col=[0, 1])
    df_prompt1 = patch_external_metrics(df_prompt1)
    df_prompt2 = pd.read_csv(prompt2_path, header=[0, 1, 2, 3], index_col=[0, 1])
    df_prompt2 = patch_external_metrics(df_prompt2)
    
    df_prompt1 = map_serializations(df_prompt1)
    df_prompt2 = map_serializations(df_prompt2)
    
    classic_datasets = DOMEN_MAP['classic']
    df_prompt1 = df_prompt1.loc[df_prompt1.index.get_level_values('Dataset').isin(classic_datasets)]
    df_prompt2 = df_prompt2.loc[df_prompt2.index.get_level_values('Dataset').isin(classic_datasets)]
    
    configs = [
        ('NoCols-NoContext', df_prompt2, 'feat_val_mask'),
        ('Cols-NoContext', df_prompt2, 'feat_val'),
        ('NoCols-Context', df_prompt1, 'feat_val_mask'),
        ('Cols-Context', df_prompt1, 'feat_val'),
    ]
    
    models = [
        ('gpt4omini', 'GPT-4o-mini'),
        ('qwen317b', 'Qwen3-1.7B'),
        ('qwen38b', 'Qwen3-8B'),
        ('qwen314b', 'Qwen3-14B'),
    ]
    
    shot = '0'
    regime = 'gen'
    metric_mean = 'roc_auc_mean'
    metric_std = 'roc_auc_std'
    
    results = []
    
    for config_name, df, serialization in configs:

        df_filt = df.loc[df.index.get_level_values('Serialization').isin([serialization])]
        
        row_data = {'Configuration': config_name}
        
        for model_key, display_name in models:
            mean_vals = []
            std_vals = []
            
            for ds in df_filt.index.get_level_values('Dataset').unique():
                try:
                    m = df_filt.loc[(ds, serialization), (shot, regime, model_key, metric_mean)]
                    s = df_filt.loc[(ds, serialization), (shot, regime, model_key, metric_std)]
                    
                    if isinstance(m, pd.Series):
                        m = m.iloc[0]
                        s = s.iloc[0]
                    
                    if m != '-' and not pd.isna(m) and m != -999.0:
                        mean_vals.append(float(m))
                        std_vals.append(float(s))
                except (KeyError, ValueError, TypeError):
                    continue
            
            if mean_vals:
                agg_mean = np.mean(mean_vals)
                agg_std = np.sqrt(np.mean(np.array(std_vals)**2))
                row_data[model_key] = (agg_mean, agg_std)
            else:
                row_data[model_key] = None
        
        results.append(row_data)
    
    latex_lines = []
    latex_lines.append("\\begin{table}[ht]")
    latex_lines.append("\\sisetup{")
    latex_lines.append("  separate-uncertainty = true,")
    latex_lines.append("  table-align-uncertainty = true,")
    latex_lines.append("  table-figures-uncertainty = 1,")
    latex_lines.append("}")
    latex_lines.append("\\centering")
    latex_lines.append("\\small")
    latex_lines.append("\\setlength{\\tabcolsep}{8pt}")
    latex_lines.append("")
    
    col_spec = "l" + "c" * len(models)
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")
    
    header = ["Configuration"] + [display for _, display in models]
    latex_lines.append(" & ".join(header) + " \\\\")
    latex_lines.append("\\midrule")
    
    for row in results:
        cells = [row['Configuration']]
        for model_key, _ in models:
            val = row.get(model_key)
            if val is not None:
                cells.append(f"\\num{{{val[0]:.3f} \\pm {val[1]:.3f}}}")
            else:
                cells.append("--")
        latex_lines.append(" & ".join(cells) + " \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append(f"\\caption{{{escape_latex(config['caption'])}}}")
    latex_lines.append(f"\\label{{{config['label']}}}")
    latex_lines.append("\\end{table}")
    
    latex_content = "\n".join(latex_lines)
    
    file_path = Path(config['tables_path']) / 'RQ2_context_configs' / "classic_domains.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_all_formats(latex_content, file_path, config)
    
    return latex_content

def get_table_models_vs_shots(config):
    
    script_dir = Path(__file__).parent
    prompt1_path = script_dir.parent / "datasets" / "agr_all_prompt1.csv"
    
    df = pd.read_csv(prompt1_path, header=[0, 1, 2, 3], index_col=[0, 1])
    df = patch_external_metrics(df)
    
    df = map_serializations(df)
    
    classic_datasets = DOMEN_MAP['classic']
    df_filt = df.loc[df.index.get_level_values('Dataset').isin(classic_datasets)]
    df_filt = df_filt.loc[df_filt.index.get_level_values('Serialization').isin(['feat_val'])]
    
    models = [
        ('gpt4omini', 'GPT-4o-mini'),
        ('qwen317b', 'Qwen3-1.7B'),
        ('qwen38b', 'Qwen3-8B'),
        ('qwen314b', 'Qwen3-14B'),
    ]
    
    shots = ['0', '4', '8', '16', '32', '64']
    regime = 'gen'
    metric_mean = 'roc_auc_mean'
    metric_std = 'roc_auc_std'
    
    results = []
    
    for model_key, display_name in models:
        row_data = {'Model': display_name}
        
        for shot in shots:
            mean_vals = []
            std_vals = []
            
            for ds in df_filt.index.get_level_values('Dataset').unique():
                try:
                    m = df_filt.loc[(ds, 'feat_val'), (shot, regime, model_key, metric_mean)]
                    s = df_filt.loc[(ds, 'feat_val'), (shot, regime, model_key, metric_std)]
                    
                    if isinstance(m, pd.Series):
                        m = m.iloc[0]
                        s = s.iloc[0]
                    
                    if m != '-' and not pd.isna(m) and m != -999.0:
                        mean_vals.append(float(m))
                        std_vals.append(float(s))
                except (KeyError, ValueError, TypeError):
                    continue
            
            if mean_vals:
                agg_mean = np.mean(mean_vals)
                agg_std = np.sqrt(np.mean(np.array(std_vals)**2))
                row_data[shot] = (agg_mean, agg_std)
            else:
                row_data[shot] = None
        
        results.append(row_data)
    
    latex_lines = []
    latex_lines.append("\\begin{table}[ht]")
    latex_lines.append("\\sisetup{")
    latex_lines.append("  separate-uncertainty = true,")
    latex_lines.append("  table-align-uncertainty = true,")
    latex_lines.append("  table-figures-uncertainty = 1,")
    latex_lines.append("}")
    latex_lines.append("\\centering")
    latex_lines.append("\\small")
    latex_lines.append("\\setlength{\\tabcolsep}{8pt}")
    latex_lines.append("")
    
    col_spec = "l" + "c" * len(shots)
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")
    
    header = ["Model"] + [f"{s}-shot" for s in shots]
    latex_lines.append(" & ".join(header) + " \\\\")
    latex_lines.append("\\midrule")
    
    for row in results:
        cells = [row['Model']]
        for shot in shots:
            val = row.get(shot)
            if val is not None:
                cells.append(f"\\num{{{val[0]:.3f} \\pm {val[1]:.3f}}}")
            else:
                cells.append("--")
        latex_lines.append(" & ".join(cells) + " \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append(f"\\caption{{{escape_latex(config['caption'])}}}")
    latex_lines.append(f"\\label{{{config['label']}}}")
    latex_lines.append("\\end{table}")
    
    latex_content = "\n".join(latex_lines)
    
    file_path = Path(config['tables_path']) / 'RQ3_models_vs_shots' / "classic_domains.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_all_formats(latex_content, file_path, config)
    
    return latex_content

def get_table_delta_context_vs_nocontext(config):

    
    script_dir = Path(__file__).parent
    prompt1_path = script_dir.parent / "datasets" / "agr_all_prompt1.csv"
    prompt2_path = script_dir.parent / "datasets" / "agr_all_prompt2.csv"

    df_prompt1 = pd.read_csv(prompt1_path, header=[0, 1, 2, 3], index_col=[0, 1])
    df_prompt1 = patch_external_metrics(df_prompt1)
    df_prompt2 = pd.read_csv(prompt2_path, header=[0, 1, 2, 3], index_col=[0, 1])
    df_prompt2 = patch_external_metrics(df_prompt2)
    
    df_prompt1 = map_serializations(df_prompt1)
    df_prompt2 = map_serializations(df_prompt2)
    
    classic_datasets = DOMEN_MAP['classic']
    df_prompt1 = df_prompt1.loc[df_prompt1.index.get_level_values('Dataset').isin(classic_datasets)]
    df_prompt2 = df_prompt2.loc[df_prompt2.index.get_level_values('Dataset').isin(classic_datasets)]
    
    df_cols_context = df_prompt1.loc[df_prompt1.index.get_level_values('Serialization').isin(['feat_val'])]
    df_nocols_nocontext = df_prompt2.loc[df_prompt2.index.get_level_values('Serialization').isin(['feat_val_mask'])]
    
    models = [
        ('gpt4omini', 'GPT-4o-mini'),
        ('qwen317b', 'Qwen3-1.7B'),
        ('qwen38b', 'Qwen3-8B'),
        ('qwen314b', 'Qwen3-14B'),
    ]
    
    shots = ['0', '4', '8', '16', '32', '64']
    regime = 'gen'
    metric_mean = 'roc_auc_mean'
    metric_std = 'roc_auc_std'
    
    results = []
    
    ds_cols = set(df_cols_context.index.get_level_values('Dataset').unique())
    ds_nocols = set(df_nocols_nocontext.index.get_level_values('Dataset').unique())
    common_datasets = ds_cols.intersection(ds_nocols)
    
    for model_key, display_name in models:
        row_data = {'Model': display_name}
        
        for shot in shots:
            deltas = []
            pooled_stds = []
            
            for ds in common_datasets:
                try:
    
                    m1 = df_cols_context.loc[(ds, 'feat_val'), (shot, regime, model_key, metric_mean)]
                    s1 = df_cols_context.loc[(ds, 'feat_val'), (shot, regime, model_key, metric_std)]
                    

                    m2 = df_nocols_nocontext.loc[(ds, 'feat_val_mask'), (shot, regime, model_key, metric_mean)]
                    s2 = df_nocols_nocontext.loc[(ds, 'feat_val_mask'), (shot, regime, model_key, metric_std)]
                    
                    if isinstance(m1, pd.Series):
                        m1 = m1.iloc[0]
                        s1 = s1.iloc[0]
                    if isinstance(m2, pd.Series):
                        m2 = m2.iloc[0]
                        s2 = s2.iloc[0]
                    
                    if (m1 != '-' and not pd.isna(m1) and m1 != -999.0 and
                        m2 != '-' and not pd.isna(m2) and m2 != -999.0):
                        delta = float(m1) - float(m2)
                        deltas.append(delta)
                        pooled_stds.append(np.sqrt(float(s1)**2 + float(s2)**2))
                        
                except (KeyError, ValueError, TypeError):
                    continue
            
            if deltas:
                mean_delta = np.mean(deltas)
                mean_pooled_std = np.sqrt(np.mean(np.array(pooled_stds)**2))
                row_data[shot] = (mean_delta, mean_pooled_std)
            else:
                row_data[shot] = None
        
        results.append(row_data)

    latex_lines = []
    latex_lines.append("\\begin{table}[ht]")
    latex_lines.append("\\sisetup{")
    latex_lines.append("  separate-uncertainty = true,")
    latex_lines.append("  table-align-uncertainty = true,")
    latex_lines.append("  table-figures-uncertainty = 1,")
    latex_lines.append("}")
    latex_lines.append("\\centering")
    latex_lines.append("\\small")
    latex_lines.append("\\setlength{\\tabcolsep}{8pt}")
    latex_lines.append("")
    
    col_spec = "l" + "c" * len(shots)
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")
    
    header = ["Model"] + [f"{s}-shot" for s in shots]
    latex_lines.append(" & ".join(header) + " \\\\")
    latex_lines.append("\\midrule")
    
    for row in results:
        cells = [row['Model']]
        for shot in shots:
            val = row.get(shot)
            if val is not None:
                sign = '+' if val[0] >= 0 else ''
                cells.append(f"\\num{{{sign}{val[0]:.3f} \\pm {val[1]:.3f}}}")
            else:
                cells.append("--")
        latex_lines.append(" & ".join(cells) + " \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append(f"\\caption{{{escape_latex(config['caption'])}}}")
    latex_lines.append(f"\\label{{{config['label']}}}")
    latex_lines.append("\\end{table}")
    
    latex_content = "\n".join(latex_lines)
    
    file_path = Path(config['tables_path']) / 'RQ4_delta_context_vs_nocontext' / "classic_domains.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_all_formats(latex_content, file_path, config)
    
    return latex_content


def get_table_zero_shot_by_dataset(config):


    script_dir = Path(__file__).parent
    prompt1_path = script_dir.parent / "datasets" / "agr_all_prompt1.csv"
    
    df = pd.read_csv(prompt1_path, header=[0, 1, 2, 3], index_col=[0, 1])
    df = patch_external_metrics(df)
    
    df = map_serializations(df)
    
    classic_datasets = DOMEN_MAP['classic']
    df_filt = df.loc[df.index.get_level_values('Dataset').isin(classic_datasets)]
    df_filt = df_filt.loc[df_filt.index.get_level_values('Serialization').isin(['feat_val'])]
    
    classic_base = [
        'hiring', 'adult', 'seismic_bumps', 'san_francisco_crimes',
        'tech_mental_health_survey', 'bbbp', 'audit', 'fraud',
        'telco', 'pc4', 'tae', 'irish', 'compas', 'vote',
        'cancer', 'steel', 'kc1', 'credit', 'transfusion',
        'fitness', 'diabetes', 'biodegr', 'marketing', 'student_famsup'
    ]
    
    def get_base_name(full_name):
        if full_name.endswith('_ls_qwen'):
            return full_name[:-8]
        elif full_name.endswith('_ls_gpt'):
            return full_name[:-7]
        return full_name
    
    models = [
        ('gpt4omini', 'GPT-4o-mini'),
        ('qwen317b', 'Qwen3-1.7B'),
        ('qwen38b', 'Qwen3-8B'),
        ('qwen314b', 'Qwen3-14B'),
    ]
    
    shot = '0'
    regime = 'gen'
    metric_mean = 'roc_auc_mean'
    metric_std = 'roc_auc_std'
    
    results = []
    
    for base_ds in classic_base:
        row_data = {'Dataset': base_ds}
        
        ds_variants = []
        for ds in df_filt.index.get_level_values('Dataset').unique():
            if get_base_name(ds) == base_ds:
                ds_variants.append(ds)
        
        for model_key, display_name in models:
            mean_vals = []
            std_vals = []
            
            for ds in ds_variants:
                try:
                    m = df_filt.loc[(ds, 'feat_val'), (shot, regime, model_key, metric_mean)]
                    s = df_filt.loc[(ds, 'feat_val'), (shot, regime, model_key, metric_std)]
                    
                    if isinstance(m, pd.Series):
                        m = m.iloc[0]
                        s = s.iloc[0]
                    
                    if m != '-' and not pd.isna(m) and m != -999.0:
                        mean_vals.append(float(m))
                        std_vals.append(float(s))
                except (KeyError, ValueError, TypeError):
                    continue
            
            if mean_vals:
   
                agg_mean = np.mean(mean_vals)
                agg_std = np.sqrt(np.mean(np.array(std_vals)**2))
                row_data[model_key] = (agg_mean, agg_std)
            else:
                row_data[model_key] = None
        
        results.append(row_data)
    
    results.sort(key=lambda x: x['Dataset'])
    
    latex_lines = []
    latex_lines.append("\\begin{table}[ht]")
    latex_lines.append("\\sisetup{")
    latex_lines.append("  separate-uncertainty = true,")
    latex_lines.append("  table-align-uncertainty = true,")
    latex_lines.append("  table-figures-uncertainty = 1,")
    latex_lines.append("}")
    latex_lines.append("\\centering")
    latex_lines.append("\\small")
    latex_lines.append("\\setlength{\\tabcolsep}{6pt}")
    latex_lines.append("")
    
    col_spec = "l" + "c" * len(models)
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")
    
    header = ["Dataset"] + [display for _, display in models]
    latex_lines.append(" & ".join(header) + " \\\\")
    latex_lines.append("\\midrule")
    
    for row in results:
        cells = [escape_latex(row['Dataset'])]
        for model_key, _ in models:
            val = row.get(model_key)
            if val is not None:
                cells.append(f"\\num{{{val[0]:.3f} \\pm {val[1]:.3f}}}")
            else:
                cells.append("--")
        latex_lines.append(" & ".join(cells) + " \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append(f"\\caption{{{escape_latex(config['caption'])}}}")
    latex_lines.append(f"\\label{{{config['label']}}}")
    latex_lines.append("\\end{table}")
    
    latex_content = "\n".join(latex_lines)
    
    file_path = Path(config['tables_path']) / 'RQ0_zero_shot_by_dataset' / "classic_datasets.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_all_formats(latex_content, file_path, config)
    
    return latex_content


def get_table_synthetic_models_shots(config):

    script_dir = Path(__file__).parent
    prompt1_path = script_dir.parent / "datasets" / "agr_all_prompt1.csv"
    
    df = pd.read_csv(prompt1_path, header=[0, 1, 2, 3], index_col=[0, 1])
    df = patch_external_metrics(df)
    
    df = map_serializations(df)
    
    synthetic_datasets = DOMEN_MAP['synthetic']
    df_filt = df.loc[df.index.get_level_values('Dataset').isin(synthetic_datasets)]
    df_filt = df_filt.loc[df_filt.index.get_level_values('Serialization').isin(['feat_val_mask'])]
    
    models = [
        ('gpt4omini', 'GPT-4o-mini'),
        ('qwen317b', 'Qwen3-1.7B'),
        ('qwen38b', 'Qwen3-8B'),
        ('qwen314b', 'Qwen3-14B'),
    ]
    
    shots = ['0', '4', '16']
    regime = 'gen'
    metric_mean = 'roc_auc_mean'
    metric_std = 'roc_auc_std'
    
    synthetic_base = [
        'mlp0_f5_h0_no_noise_ReLU',
        'mlp1_f10_h10_no_noise_ReLU',
        'mlp2_f15_h15_no_noise_ReLU',
        'mlp3_f20_h20_no_noise_ReLU',
        'mlp5_f30_h30_no_noise_ReLU',
        'mlp7_f40_h40_no_noise_ReLU',
        'mlp9_f50_h50_no_noise_ReLU',
    ]
    
    synthetic_display = ['MLP0', 'MLP1', 'MLP2', 'MLP3', 'MLP5', 'MLP7', 'MLP9']
    
    results = []
    
    for base_ds, display_ds in zip(synthetic_base, synthetic_display):
        row_data = {'Dataset': display_ds}
        
        for model_key, display_name in models:
            for shot in shots:
                col_key = f'{model_key}_{shot}'
                
                try:
                    m = df_filt.loc[(base_ds, 'feat_val_mask'), (shot, regime, model_key, metric_mean)]
                    s = df_filt.loc[(base_ds, 'feat_val_mask'), (shot, regime, model_key, metric_std)]
                    
                    if isinstance(m, pd.Series):
                        m = m.iloc[0]
                        s = s.iloc[0]
                    
                    if m != '-' and not pd.isna(m) and m != -999.0:
                        row_data[col_key] = (float(m), float(s))
                    else:
                        row_data[col_key] = None
                except (KeyError, ValueError, TypeError):
                    row_data[col_key] = None
        
        results.append(row_data)
    
    latex_lines = []
    latex_lines.append("\\begin{table}[ht]")
    latex_lines.append("\\sisetup{")
    latex_lines.append("  separate-uncertainty = true,")
    latex_lines.append("  table-align-uncertainty = true,")
    latex_lines.append("  table-figures-uncertainty = 1,")
    latex_lines.append("}")
    latex_lines.append("\\centering")
    latex_lines.append("\\small")
    latex_lines.append("\\setlength{\\tabcolsep}{4pt}")
    latex_lines.append("")
    
    num_cols = len(models) * len(shots)
    col_spec = "l" + "c" * num_cols
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")
    
    header1 = ["Dataset"]
    for model_key, display_name in models:
        header1.append(f"\\multicolumn{{{len(shots)}}}{{c}}{{{display_name}}}")
    latex_lines.append(" & ".join(header1) + " \\\\")
    
    header2 = [""]
    for model_key, display_name in models:
        for shot in shots:
            header2.append(f"{shot}-shot")
    latex_lines.append(" & ".join(header2) + " \\\\")
    latex_lines.append("\\midrule")
    
    for row in results:
        cells = [row['Dataset']]
        for model_key, _ in models:
            for shot in shots:
                val = row.get(f'{model_key}_{shot}')
                if val is not None:
                    cells.append(f"\\num{{{val[0]:.3f} \\pm {val[1]:.3f}}}")
                else:
                    cells.append("--")
        latex_lines.append(" & ".join(cells) + " \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append(f"\\caption{{{escape_latex(config['caption'])}}}")
    latex_lines.append(f"\\label{{{config['label']}}}")
    latex_lines.append("\\end{table}")
    
    latex_content = "\n".join(latex_lines)
    
    file_path = Path(config['tables_path']) / 'RQ6_synthetic_models_shots' / "synthetic_datasets.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_all_formats(latex_content, file_path, config)
    
    return latex_content

def get_table_icl_combinations(config):

    script_dir = Path(__file__).parent
    prompt1_path = script_dir.parent / "datasets" / "agr_all_prompt1.csv"
    prompt2_path = script_dir.parent / "datasets" / "agr_all_prompt2.csv"
    prompt3_path = script_dir.parent / "datasets" / "agr_all_prompt3.csv"

    df1 = pd.read_csv(prompt1_path, header=[0, 1, 2, 3], index_col=[0, 1])
    df1 = patch_external_metrics(df1)
    df2 = pd.read_csv(prompt2_path, header=[0, 1, 2, 3], index_col=[0, 1])
    df2 = patch_external_metrics(df2)
    df3 = pd.read_csv(prompt3_path, header=[0, 1, 2, 3], index_col=[0, 1])
    df3 = patch_external_metrics(df3)
    
    df1 = map_serializations(df1)
    df2 = map_serializations(df2)
    df3 = map_serializations(df3)

    all_datasets_all_domains = []
    for domain_datasets in DOMEN_MAP.values():
        all_datasets_all_domains.extend(domain_datasets)

    llm_synth_qwen = [ds for ds in all_datasets_all_domains if ds.endswith('_ls_qwen')]
    llm_synth_gpt = [ds for ds in all_datasets_all_domains if ds.endswith('_ls_gpt')]
    
    model_dataset_map = {
        'qwen317b': llm_synth_qwen,
        'qwen38b': llm_synth_qwen,
        'qwen314b': llm_synth_qwen,
        'gpt4omini': llm_synth_gpt,
    }
    
    models = [
        ('qwen317b', 'Qwen3-1.7B'),
        ('qwen38b', 'Qwen3-8B'),
        ('qwen314b', 'Qwen3-14B'),
        ('gpt4omini', 'GPT-4o-mini'),
    ]

    configs = [
        ('Few-shot (16)', df2, '16', 'gen', 'feat_val_mask'),
        ('Expert', df3, '0', 'gen', 'feat_val_mask'),
        ('Prior', df1, '0', 'gen', 'feat_val'),
        ('Few-shot (16) + expert', df3, '16', 'gen', 'feat_val_mask'),
        ('Few-shot (16) + prior', df1, '16', 'gen', 'feat_val'),
        ('Expert + prior', df3, '0', 'gen', 'feat_val'),
        ('Few-shot (16) + expert + prior', df3, '16', 'gen', 'feat_val'),
    ]
    
    metric_mean = 'roc_auc_mean'
    metric_std = 'roc_auc_std'
    
    results = []
    
    for config_name, df, shot, regime, serialization in configs:
        row_data = {'Configuration': config_name}
        
        for model_key, display_name in models:
            datasets = model_dataset_map[model_key]
            mean_vals = []
            std_vals = []
            
            for ds in datasets:
                try:
                    m = df.loc[(ds, serialization), (shot, regime, model_key, metric_mean)]
                    s = df.loc[(ds, serialization), (shot, regime, model_key, metric_std)]
                    
                    if isinstance(m, pd.Series):
                        m = m.iloc[0]
                        s = s.iloc[0]
                    
                    if m != '-' and not pd.isna(m) and m != -999.0:
                        mean_vals.append(float(m))
                        std_vals.append(float(s))
                except (KeyError, ValueError, TypeError):
                    continue
            
            if mean_vals:
                agg_mean = np.mean(mean_vals)
                agg_std = np.sqrt(np.mean(np.array(std_vals)**2))
                row_data[model_key] = (agg_mean, agg_std)
            else:
                row_data[model_key] = None
        
        results.append(row_data)
    
    tabpfn_row = {'Configuration': 'TabPFN (16)'}
    for model_key, display_name in models:
        datasets = model_dataset_map[model_key]
        mean_vals = []
        std_vals = []
        
        for ds in datasets:
            try:
                m = df1.loc[(ds, 'feat_val'), ('16', 'nogen', 'tabpfn', metric_mean)]
                s = df1.loc[(ds, 'feat_val'), ('16', 'nogen', 'tabpfn', metric_std)]
                
                if isinstance(m, pd.Series):
                    m = m.iloc[0]
                    s = s.iloc[0]
                
                if m != '-' and not pd.isna(m) and m != -999.0:
                    mean_vals.append(float(m))
                    std_vals.append(float(s))
            except (KeyError, ValueError, TypeError):
                continue
        
        if mean_vals:
            agg_mean = np.mean(mean_vals)
            agg_std = np.sqrt(np.mean(np.array(std_vals)**2))
            tabpfn_row[model_key] = (agg_mean, agg_std)
        else:
            tabpfn_row[model_key] = None
    
    results.append(tabpfn_row)
    
    local_max = {}
    for model_key, _ in models:
        vals = []
        for i in range(3):
            val = results[i].get(model_key)
            if val is not None:
                vals.append((i, val[0]))
        if vals:
            local_max[model_key] = max(vals, key=lambda x: x[1])[0]
    
    global_max = {}
    for model_key, _ in models:
        vals = []
        for i in range(7):
            val = results[i].get(model_key)
            if val is not None:
                vals.append((i, val[0]))
        if vals:
            global_max[model_key] = max(vals, key=lambda x: x[1])[0]
    
    latex_lines = []
    latex_lines.append("\\begin{table*}[ht]")
    latex_lines.append("\\sisetup{")
    latex_lines.append("  separate-uncertainty = true,")
    latex_lines.append("  table-align-uncertainty = true,")
    latex_lines.append("  table-figures-uncertainty = 1,")
    latex_lines.append("}")
    latex_lines.append("\\centering")
    latex_lines.append("\\small")
    latex_lines.append("\\setlength{\\tabcolsep}{6pt}")
    latex_lines.append("")
    
    col_spec = "l" + "c" * len(models)
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")
    
    header = ["\\textbf{Prompting Regime}"] + [f"\\textbf{{{display}}}" for _, display in models]
    latex_lines.append(" & ".join(header) + " \\\\")
    latex_lines.append("\\midrule")
    
    for idx, row in enumerate(results):
        cells = [row['Configuration']]
        
        for model_key, _ in models:
            val = row.get(model_key)
            if val is None:
                cells.append("--")
                continue
            
            v_mean, v_std = val
            cell_content = f"\\num{{{v_mean:.3f} \\pm {v_std:.3f}}}"
            
            if idx < 3 and idx == local_max.get(model_key, -1):
                cell_content = f"\\textbf{{{cell_content}}}"
            
            if idx < 7 and idx == global_max.get(model_key, -1):
                cell_content = f"\\underline{{{cell_content}}}"
            
            cells.append(cell_content)
        
        latex_lines.append(" & ".join(cells) + " \\\\")
        
        if idx == 2:
            latex_lines.append("\\hlineB{2}")
        elif idx == 6:
            latex_lines.append("\\hlineB{2}")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append(f"\\caption{{{escape_latex(config['caption'])}}}")
    latex_lines.append(f"\\label{{{config['label']}}}")
    latex_lines.append("\\end{table*}")
    
    latex_content = "\n".join(latex_lines)
    
    file_path = Path(config['tables_path']) / 'RQ5_icl_combinations' / "llm_synthetic.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_all_formats(latex_content, file_path, config)
    
    return latex_content

def get_table_new_datasets_zero_shot(config):

    script_dir = Path(__file__).parent
    prompt1_path = script_dir.parent / "datasets" / "agr_all_prompt1.csv"
    featllm_path = script_dir.parent / "datasets" / "new_datasets_FeatLLM.csv"
    llm_trees_path = script_dir.parent / "datasets" / "new_datasets_llm_trees.csv"
    
    df = pd.read_csv(prompt1_path, header=[0, 1, 2, 3], index_col=[0, 1])
    df = patch_external_metrics(df)
    df_featllm = pd.read_csv(featllm_path)
    df_llm_trees = pd.read_csv(llm_trees_path)
    
    featllm_data = {}
    for _, row in df_featllm.iterrows():
        ds_name = row['dataset']
        if 'shot_0' in row and not pd.isna(row['shot_0']):
            val_str = str(row['shot_0'])
            match = re.match(r'([\d.]+)\s*±\s*([\d.]+)', val_str)
            if match:
                featllm_data[ds_name] = (float(match.group(1)), float(match.group(2)))
    
    llm_trees_data = {}
    for _, row in df_llm_trees.iterrows():
        ds_name = row['dataset']
        if 'rocauc_avg' in row and 'rocauc_std' in row:
            if not pd.isna(row['rocauc_avg']) and not pd.isna(row['rocauc_std']):
                llm_trees_data[ds_name] = (float(row['rocauc_avg']), float(row['rocauc_std']))
    
    df = map_serializations(df)
    
    new_datasets = DOMEN_MAP['new']
    df_filt = df.loc[df.index.get_level_values('Dataset').isin(new_datasets)]
    df_filt = df_filt.loc[df_filt.index.get_level_values('Serialization').isin(['feat_val'])]
    
    new_base = [
        'stars', 'machine', 'callcenter', 'bank_credit_scoring',
        'extrovert', 'reading', 'postpartum', 'crimes_arrest'
    ]
    
    def get_base_name(full_name):

        if full_name.endswith('_ls_qwen'):
            return full_name[:-8]
        elif full_name.endswith('_ls_gpt'):
            return full_name[:-7]
        return full_name
    
    models = [
        ('gpt4omini', 'GPT-4o-mini'),
        ('qwen317b', 'Qwen3-1.7B'),
        ('qwen38b', 'Qwen3-8B'),
        ('qwen314b', 'Qwen3-14B'),
    ]
    
    shot = '0'
    regime = 'gen'
    metric_mean = 'roc_auc_mean'
    metric_std = 'roc_auc_std'
    
    results = []
    
    for base_ds in new_base:
        row_data = {'Dataset': base_ds}
        
        ds_variants = []
        for ds in df_filt.index.get_level_values('Dataset').unique():
            if get_base_name(ds) == base_ds:
                ds_variants.append(ds)
        
        for model_key, display_name in models:
            mean_vals = []
            std_vals = []
            
            for ds in ds_variants:
                try:
                    m = df_filt.loc[(ds, 'feat_val'), (shot, regime, model_key, metric_mean)]
                    s = df_filt.loc[(ds, 'feat_val'), (shot, regime, model_key, metric_std)]
                    
                    if isinstance(m, pd.Series):
                        m = m.iloc[0]
                        s = s.iloc[0]
                    
                    if m != '-' and not pd.isna(m) and m != -999.0:
                        mean_vals.append(float(m))
                        std_vals.append(float(s))
                except (KeyError, ValueError, TypeError):
                    continue
            
            if mean_vals:
                agg_mean = np.mean(mean_vals)
                agg_std = np.sqrt(np.mean(np.array(std_vals)**2))
                row_data[model_key] = (agg_mean, agg_std)
            else:
                row_data[model_key] = None
        
        if base_ds in featllm_data:
            row_data['FeatLLM'] = featllm_data[base_ds]
        else:
            row_data['FeatLLM'] = None
        
        if base_ds in llm_trees_data:
            row_data['LLM-tree'] = llm_trees_data[base_ds]
        else:
            row_data['LLM-tree'] = None
        
        results.append(row_data)
    
    latex_lines = []
    latex_lines.append("\\begin{table}[ht]")
    latex_lines.append("\\sisetup{")
    latex_lines.append("  separate-uncertainty = true,")
    latex_lines.append("  table-align-uncertainty = true,")
    latex_lines.append("  table-figures-uncertainty = 1,")
    latex_lines.append("}")
    latex_lines.append("\\centering")
    latex_lines.append("\\small")
    latex_lines.append("\\setlength{\\tabcolsep}{4pt}")
    latex_lines.append("")
    
    all_columns = models + [('FeatLLM', 'FeatLLM'), ('LLM-tree', 'LLM-tree')]
    col_spec = "l" + "c" * len(all_columns)
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")
    
    header = ["Dataset"] + [display for _, display in all_columns]
    latex_lines.append(" & ".join(header) + " \\\\")
    latex_lines.append("\\midrule")
    
    for row in results:
        cells = [escape_latex(row['Dataset'])]
        for model_key, _ in all_columns:
            val = row.get(model_key)
            if val is not None:
                cells.append(f"\\num{{{val[0]:.3f} \\pm {val[1]:.3f}}}")
            else:
                cells.append("--")
        latex_lines.append(" & ".join(cells) + " \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append(f"\\caption{{{escape_latex(config['caption'])}}}")
    latex_lines.append(f"\\label{{{config['label']}}}")
    latex_lines.append("\\end{table}")
    
    latex_content = "\n".join(latex_lines)
    
    file_path = Path(config['tables_path']) / 'new_datasets_zero_shot' / "new_datasets.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_all_formats(latex_content, file_path, config)
    
    return latex_content


def get_table_shots_vs_models_new_datasets(config):

    script_dir = Path(__file__).parent
    prompt1_path = script_dir.parent / "datasets" / "agr_all_prompt1.csv"
    featllm_path = script_dir.parent / "datasets" / "new_datasets_FeatLLM.csv"
    
    df = pd.read_csv(prompt1_path, header=[0, 1, 2, 3], index_col=[0, 1])
    df = patch_external_metrics(df)
    df_featllm = pd.read_csv(featllm_path)
    
    featllm_data = {}
    for _, row in df_featllm.iterrows():
        ds_name = row['dataset']
        featllm_data[ds_name] = {}
        for shot in ['0', '4', '8', '16', '32', '64']:
            col_name = f'shot_{shot}'
            if col_name in row and not pd.isna(row[col_name]):
                val_str = str(row[col_name])
                match = re.match(r'([\d.]+)\s*±\s*([\d.]+)', val_str)
                if match:
                    featllm_data[ds_name][shot] = (float(match.group(1)), float(match.group(2)))
    
    df = map_serializations(df)
    
    new_datasets = DOMEN_MAP['new']
    df_filt = df.loc[df.index.get_level_values('Dataset').isin(new_datasets)]
    df_filt = df_filt.loc[df_filt.index.get_level_values('Serialization').isin(['feat_val'])]
    
    models = [
        ('gpt4omini', 'GPT-4o-mini'),
        ('qwen317b', 'Qwen3-1.7B'),
        ('qwen38b', 'Qwen3-8B'),
        ('qwen314b', 'Qwen3-14B'),
    ]
    
    shots = ['0', '4', '8', '16', '32', '64']
    regime = 'gen'
    metric_mean = 'roc_auc_mean'
    metric_std = 'roc_auc_std'
    
    model_data = {}
    for model_key, display_name in models:
        model_data[model_key] = {}
        for shot in shots:
            mean_vals = []
            std_vals = []
            
            for ds in df_filt.index.get_level_values('Dataset').unique():
                try:
                    m = df_filt.loc[(ds, 'feat_val'), (shot, regime, model_key, metric_mean)]
                    s = df_filt.loc[(ds, 'feat_val'), (shot, regime, model_key, metric_std)]
                    
                    if isinstance(m, pd.Series):
                        m = m.iloc[0]
                        s = s.iloc[0]
                    
                    if m != '-' and not pd.isna(m) and m != -999.0:
                        mean_vals.append(float(m))
                        std_vals.append(float(s))
                except (KeyError, ValueError, TypeError):
                    continue
            
            if mean_vals:
                agg_mean = np.mean(mean_vals)
                agg_std = np.sqrt(np.mean(np.array(std_vals)**2))
                model_data[model_key][shot] = (agg_mean, agg_std)
            else:
                model_data[model_key][shot] = None
    
    featllm_by_shot = {}
    for shot in shots:
        featllm_means = []
        featllm_stds = []
        for ds_name, ds_data in featllm_data.items():
            if shot in ds_data:
                m, s = ds_data[shot]
                featllm_means.append(m)
                featllm_stds.append(s)
        
        if featllm_means:
            agg_mean = np.mean(featllm_means)
            agg_std = np.sqrt(np.mean(np.array(featllm_stds)**2))
            featllm_by_shot[shot] = (agg_mean, agg_std)
        else:
            featllm_by_shot[shot] = None
    
    results = []
    
    for model_key, display_name in models:
        row_data = {'Model': display_name}
        for shot in shots:
            row_data[shot] = model_data[model_key][shot]
        results.append(row_data)
    
    featllm_row = {'Model': 'FeatLLM'}
    for shot in shots:
        featllm_row[shot] = featllm_by_shot[shot]
    results.append(featllm_row)
    
    latex_lines = []
    latex_lines.append("\\begin{table}[ht]")
    latex_lines.append("\\sisetup{")
    latex_lines.append("  separate-uncertainty = true,")
    latex_lines.append("  table-align-uncertainty = true,")
    latex_lines.append("  table-figures-uncertainty = 1,")
    latex_lines.append("}")
    latex_lines.append("\\centering")
    latex_lines.append("\\small")
    latex_lines.append("\\setlength{\\tabcolsep}{6pt}")
    latex_lines.append("")
    
    col_spec = "l" + "c" * len(shots)
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")
    
    header = ["Model"] + [f"{s}-shot" for s in shots]
    latex_lines.append(" & ".join(header) + " \\\\")
    latex_lines.append("\\midrule")
    
    for row in results:
        cells = [row['Model']]
        for shot in shots:
            val = row.get(shot)
            if val is not None:
                cells.append(f"\\num{{{val[0]:.3f} \\pm {val[1]:.3f}}}")
            else:
                cells.append("--")
        latex_lines.append(" & ".join(cells) + " \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append(f"\\caption{{{escape_latex(config['caption'])}}}")
    latex_lines.append(f"\\label{{{config['label']}}}")
    latex_lines.append("\\end{table}")
    
    latex_content = "\n".join(latex_lines)
    
    file_path = Path(config['tables_path']) / 'models_vs_shots_new' / "new_datasets.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_all_formats(latex_content, file_path, config)
    
    return latex_content