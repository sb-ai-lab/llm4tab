# Utility function for getting data & prompting & query
import os
import random
from openai import OpenAI
import time
import torch
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

TASK_DICT = {
    'blood': "Did the person donate blood? Yes or no?",
    'credit-g': "Does this person receive a credit? Yes or no?",
    'diabetes': "Does this patient have diabetes? Yes or no?",
    'heart': "Does the coronary angiography of this patient show a heart disease? Yes or no?",
    'adult': "Does this person earn more than 50000 dollars per year? Yes or no?",
    'bank': "Does this client subscribe to a term deposit? Yes or no?",
    'car': "How would you rate the decision to buy this car? Unacceptable, acceptable, good or very good?",
    'communities': "How high will the rate of violent crimes per 100K population be in this area. Low, medium, or high?",
    'myocardial': "Does the myocardial infarction complications data of this patient show chronic heart failure? Yes or no?",
    'bank_credit_scoring': "Does this bank client have high credit scoring? Yes or no?",
    'callcenter': "Was this client satisfied with the callcenter work? Yes or no?",
    'postpartum': "Does this hospital female patient have postpartum depression? Yes or no?",
    'machine': "Is this industrial machine at risk of failure? Yes or no?",
    'stars': "Is this star a giant star? Yes or no?",
    'crimes_arrest': "Was an arrest made for this reported crime in Chicago? Yes or no?",
    'reading': "Does this undergraduate student read academic books most of the time? Yes or no?",
    'extrovert': "Does this person maintain long-distance friendships through calls? Yes or no?"
}



def evaluate(pred_probs, answers, multiclass=False):   
    if multiclass == False:
        result_auc = roc_auc_score(answers, pred_probs[:, 1])
    else:
        result_auc = roc_auc_score(answers, pred_probs, multi_class='ovr', average='macro')        
    return result_auc


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_dataset(data_name, shot, seed):

    file_name = f"./data/{data_name}.csv"
    df = pd.read_csv(file_name)

    if 'Unnamed: 0' in df.columns:
         df = df.drop(columns=['Unnamed: 0'])

    default_target_attribute = df.columns[-1]
    
    categorical_indicator = [True if (dt == np.dtype('O') or pd.api.types.is_string_dtype(dt)) else False for dt in df.dtypes.tolist()][:-1]
    attribute_names = df.columns[:-1].tolist()

    X = df.convert_dtypes()
    y = df[default_target_attribute].to_numpy()
    label_list = np.unique(y).tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X.drop(default_target_attribute, axis=1),
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )
    
    assert(shot <= 128) # We only consider the low-shot regimes here
    X_new_train = X_train.copy()
    X_new_train[default_target_attribute] = y_train
    sampled_list = []
    total_shot_count = 0
    remainder = shot % len(np.unique(y_train))
    for _, grouped in X_new_train.groupby(default_target_attribute):
        sample_num = shot // len(np.unique(y_train))
        if remainder > 0:
            sample_num += 1
            remainder -= 1
        grouped = grouped.sample(sample_num, random_state=seed)
        sampled_list.append(grouped)
    X_balanced = pd.concat(sampled_list)
    X_train = X_balanced.drop([default_target_attribute], axis=1)
    y_train = X_balanced[default_target_attribute].to_numpy()

    return df, X_train, X_test, y_train, y_test, default_target_attribute, label_list, categorical_indicator


def get_dataset_TabLLMBench(data_name, shot, seed, use_custom_split=True, ratio='50/50'):

    file_name = f"./data/{data_name}.csv"
    df = pd.read_csv(file_name)

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    default_target_attribute = df.columns[-1]
    
    categorical_indicator = [True if (dt == np.dtype('O') or pd.api.types.is_string_dtype(dt)) else False for dt in df.dtypes.tolist()][:-1]
    attribute_names = df.columns[:-1].tolist()

    X = df.drop(columns=[default_target_attribute])
    y = df[default_target_attribute].to_numpy()
    label_list = np.unique(y).tolist()
    
    if use_custom_split:
        rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=42)
        train_idx, test_idx = next(rskf.split(X, y))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
            stratify=y
        )
    
    assert(shot <= 128)
    
    X_new_train = X_train.copy()
    X_new_train[default_target_attribute] = y_train
    
    df_train = X_new_train.copy()
    df_train['label'] = df_train[default_target_attribute].astype(int).astype(str)
    
    train_0 = df_train[df_train['label'] == '0']
    train_1 = df_train[df_train['label'] == '1']
    
    ratio_parts = ratio.split('/')
    need_0 = int(int(shot) * float(ratio_parts[0]) / 100)
    need_1 = int(int(shot) * float(ratio_parts[1]) / 100)
    
    available_0 = len(train_0)
    available_1 = len(train_1)
    
    take_0 = min(need_0, available_0)
    take_1 = min(need_1, available_1)
    
    sampled_0 = train_0.sample(take_0, random_state=seed) if take_0 > 0 else pd.DataFrame()
    sampled_1 = train_1.sample(take_1, random_state=seed) if take_1 > 0 else pd.DataFrame()
    
    result = pd.concat([sampled_0, sampled_1]).reset_index(drop=True)

    if shot:
    
        X_train = result.drop([default_target_attribute, 'label'], axis=1)
        y_train = result[default_target_attribute].to_numpy()

    else:

        X_train = []


    return df, X_train, X_test, y_train, y_test, default_target_attribute, label_list, categorical_indicator


def query_gpt(text_list, api_key, max_tokens=30, temperature=0, max_try_num=10, model="openai/gpt-4o-mini"):

    result_list = []
    for prompt in tqdm(text_list):
        curr_try_num = 0
        while curr_try_num < max_try_num:
            try:
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                results = response.choices[0].message.content
                result_list.append(results)
                break
            except Exception as e:
                print(e)
                curr_try_num += 1
                if curr_try_num >= max_try_num:
                    result_list.append(-1)
                time.sleep(10)
    return result_list


def serialize(row):
    target_str = f""
    for attr_idx, attr_name in enumerate(list(row.index)):
        if attr_idx < len(list(row.index)) - 1:
            target_str += " is ".join([attr_name, str(row[attr_name]).strip(" .'").strip('"').strip()])
            target_str += ". "
        else:
            if len(attr_name.strip()) < 2:
                continue
            target_str += " is ".join([attr_name, str(row[attr_name]).strip(" .'").strip('"').strip()])
            target_str += "."
    return target_str


def fill_in_templates(fill_in_dict, template_str):
    for key, value in fill_in_dict.items():
        if key in template_str:
            template_str = template_str.replace(key, value)
    return template_str    


def parse_rules(result_texts, label_list=[]):
    total_rules = []
    splitter = "conditions for class"
    for text in result_texts:
        splitted = text.split(splitter)
        if splitter not in text:
            continue
        if len(label_list) != 0 and len(splitted) != len(label_list) + 1:
            continue
        
        rule_raws = splitted[1:]
        rule_dict = {}
        for rule_raw in rule_raws:
            class_name = rule_raw.split(":")[0].strip(" .'").strip(' []"')
            rule_parsed = []
            for txt in rule_raw.strip().split("\n")[1:]:
                if len(txt) < 2:
                    break
                rule_parsed.append(" ".join(txt.strip().split(" ")[1:]))
                rule_dict[class_name] = rule_parsed
        total_rules.append(rule_dict)
    return total_rules





def get_prompt_for_asking(data_name, df_all, df_x, df_y, label_list, 
                          default_target_attribute, file_name, meta_file_name, is_cat, num_query=5):
    with open(file_name, "r") as f:
        prompt_type_str = f.read()
        
    try:
        with open(meta_file_name, "r") as f:
            meta_data = json.load(f)
    except:
        meta_data = {}
    
    task_desc = f"{TASK_DICT[data_name]}\n"

    if df_x is not None and df_y is not None and len(df_x) > 0:
        df_incontext = df_x.copy()
        df_incontext[default_target_attribute] = df_y 
    else:
        df_incontext = pd.DataFrame()
    
    format_list = [f'10 different conditions for class "{label}":\n- [Condition]\n...' for label in label_list]
    format_desc = '\n\n'.join(format_list)
            
    template_list = []
    current_query_num = 0
    end_flag = False
    while True:     
        if current_query_num >= num_query:
            break
                        
        if len(df_incontext) > 0 and len(df_incontext.columns) >= 20:
            total_column_list = []
            for i in range(len(df_incontext.columns) // 10):
                column_list = df_incontext.columns.tolist()[:-1]
                random.shuffle(column_list)
                total_column_list.append(column_list[i*10:(i+1)*10])
        elif len(df_incontext) > 0:
            total_column_list = [df_incontext.columns.tolist()[:-1]]
        else:
            total_column_list = [df_all.columns.tolist()]
            
        for selected_column in total_column_list:
            if current_query_num >= num_query:
                break
                
            if len(df_incontext) > 0:
                threshold = 16   
                if len(df_incontext) > threshold:
                    sample_num = int(threshold / df_incontext[default_target_attribute].nunique())
                    df_incontext = df_incontext.groupby(
                        default_target_attribute, group_keys=False
                    ).apply(lambda x: x.sample(sample_num))
                
                feature_name_list = []
                sel_cat_idx = [df_incontext.columns.tolist().index(col_name) for col_name in selected_column if col_name != default_target_attribute]
                is_cat_sel = np.array(is_cat)[sel_cat_idx]
                
                for cidx, cname in enumerate([c for c in selected_column if c != default_target_attribute]):
                    if is_cat_sel[cidx] == True:
                        clist = df_all[cname].unique().tolist()
                        clist = [str(c) for c in clist]
                        if len(clist) > 20:
                            clist_str = f"{clist[0]}, {clist[1]}, ..., {clist[-1]}"
                        else:
                            clist_str = ", ".join(clist)
                        desc = meta_data[cname] if cname in meta_data.keys() else ""
                        feature_name_list.append(f"- {cname}: {desc} (categorical variable with categories [{clist_str}])")
                    else:
                        desc = meta_data[cname] if cname in meta_data.keys() else ""
                        feature_name_list.append(f"- {cname}: {desc} (numerical variable)")
            else:
                feature_name_list = []
                for cname in selected_column:
                    if is_cat[df_all.columns.tolist().index(cname)] == True:
                        clist = df_all[cname].unique().tolist()
                        clist = [str(c) for c in clist]
                        print(clist)
                        if len(clist) > 20:
                            clist_str = f"{clist[0]}, {clist[1]}, ..., {clist[-1]}"
                        else:
                            clist_str = ", ".join(clist)
                        desc = meta_data[cname] if cname in meta_data.keys() else ""
                        feature_name_list.append(f"- {cname}: {desc} (categorical variable with categories [{clist_str}])")
                    else:
                        desc = meta_data[cname] if cname in meta_data.keys() else ""
                        feature_name_list.append(f"- {cname}: {desc} (numerical variable)")

            feature_desc = "\n".join(feature_name_list)
            
            in_context_desc = ""  
            if len(df_incontext) > 0:
                df_current = df_incontext.copy()
                df_current = df_current.groupby(
                    default_target_attribute, group_keys=False
                ).apply(lambda x: x.sample(frac=1))

                for icl_idx, icl_row in df_current.iterrows():
                    answer = icl_row[default_target_attribute]
                    icl_row = icl_row.drop(labels=default_target_attribute)  
                    icl_row = icl_row[selected_column]
                    in_context_desc += serialize(icl_row)
                    in_context_desc += f"\nAnswer: {answer}\n"

            fill_in_dict = {
                "[TASK]": task_desc, 
                "[EXAMPLES]": in_context_desc,
                "[FEATURES]": feature_desc,
                "[FORMAT]": format_desc
            }
            template = fill_in_templates(fill_in_dict, prompt_type_str)
            template_list.append(template)
            current_query_num += 1
        
    return template_list, feature_desc


def get_prompt_for_generating_function(parsed_rule, feature_desc, file_name):
    with open(file_name, "r") as f:
        prompt_type_str = f.read()
    
    template_list = []
    for class_id, each_rule in parsed_rule.items():
        function_name = f'extracting_features_{class_id}'
        rule_str = '\n'.join([f'- {k}' for k in each_rule])
    
        fill_in_dict = {
            "[NAME]": function_name, 
            "[CONDITIONS]": rule_str,
            "[FEATURES]": feature_desc
        }
        template = fill_in_templates(fill_in_dict, prompt_type_str)
        template_list.append(template)
        
    return template_list


def convert_to_binary_vectors(fct_strs_all, fct_names, label_list, X_train, X_test):
    X_train_all_dict = {}
    X_test_all_dict = {}
    executable_list = []
    for i in range(len(fct_strs_all)):
        X_train_dict, X_test_dict = {}, {}
        for label in label_list:
            X_train_dict[label] = {}
            X_test_dict[label] = {}

        fct_idx_dict = {}
        for idx, name in enumerate(fct_names[i]):
            for label in label_list:
                label_name = '_'.join(label.split(' '))
                if label_name.lower() in name.lower():
                    fct_idx_dict[label] = idx

        if len(fct_idx_dict) != len(label_list):
            print(fct_idx_dict)
            continue
        try:
            for label in label_list:
                fct_idx = fct_idx_dict[label]
                exec(fct_strs_all[i][fct_idx].strip('` "'))
                
                if X_train is not None and len(X_train) > 0:
                    X_train_each = locals()[fct_names[i][fct_idx]](X_train).astype('int').to_numpy()
                    X_train_dict[label] = torch.tensor(X_train_each).float()
                
                X_test_each = locals()[fct_names[i][fct_idx]](X_test).astype('int').to_numpy()
                X_test_dict[label] = torch.tensor(X_test_each).float()
                
                if X_train is not None and len(X_train) > 0:
                    assert(X_train_each.shape[1] == X_test_each.shape[1])

            X_train_all_dict[i] = X_train_dict
            X_test_all_dict[i] = X_test_dict
            executable_list.append(i)
        except Exception:
            continue

    return executable_list, X_train_all_dict, X_test_all_dict

def get_table_shots_vs_models_new_datasets(config):

    script_dir = Path(__file__).parent
    prompt1_path = script_dir.parent / "datasets" / "agr_all_prompt_1.csv"
    featllm_path = script_dir.parent / "datasets" / "new_datasets_FeatLLM.csv"
    
    df = pd.read_csv(prompt1_path, header=[0, 1, 2, 3], index_col=[0, 1])
    df_featllm = pd.read_csv(featllm_path)
    
    featllm_data = {}
    for _, row in df_featllm.iterrows():
        ds_name = row['dataset']
        featllm_data[ds_name] = {}
        for shot in ['0', '4', '8', '16', '32', '64']:
            col_name = f'shot_{shot}'
            if col_name in row and not pd.isna(row[col_name]):
                val_str = str(row[col_name])
                # "0.697 ± 0.028" -> (0.697, 0.028)
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
    
    results = []
    
    for shot in shots:
        row_data = {'Shots': f'{shot}-shot'}
        
        for model_key, display_name in models:
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
                row_data[model_key] = (agg_mean, agg_std)
            else:
                row_data[model_key] = None
        
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
            row_data['FeatLLM'] = (agg_mean, agg_std)
        else:
            row_data['FeatLLM'] = None
        
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
    latex_lines.append("\\setlength{\\tabcolsep}{6pt}")
    latex_lines.append("")
    
    all_columns = models + [('FeatLLM', 'FeatLLM')]
    col_spec = "l" + "c" * len(all_columns)
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")
    
    header = ["Shots"] + [display for _, display in all_columns]
    latex_lines.append(" & ".join(header) + " \\\\")
    latex_lines.append("\\midrule")
    
    for row in results:
        cells = [row['Shots']]
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
    
    file_path = Path(config['tables_path']) / 'shots_vs_models_new' / "new_datasets.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    return latex_content