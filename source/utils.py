import os
import numpy as np
import pandas as pd
import openml
import pickle
import inflect
from datetime import datetime
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler



def get_df(config):

    dataset_name = config['data']['DATASET_NAME']
    local_dataset_path = config['data']['LOCAL_DATASET_PATH']
    df_type = config['data']['DF_TYPE']
    df_format = config['data']['DF_FORMAT']

    if df_type == 'custom':
        X_train, y_train, X_test, y_test = get_custom_df(dataset_name, local_dataset_path, df_format)
    elif df_type == 'openml':
        X_train, y_train, X_test, y_test = get_openml_df(dataset_name, config)
    else:
        raise ValueError("Enter correct DF_TYPE parameter: 'custom', 'openml'.")

    
    return X_train, y_train, X_test, y_test
    
    

def get_custom_df(df_name, base_path, df_format):
    
    # TODO: add parquet and xlsx support - search with hash map
    if df_format == 'csv':
        path = base_path + df_name + '.csv'
        df = pd.read_csv(path, index_col=0)
    elif df_format == 'parquet':
        pass
    elif df_format == 'xlsx':
        pass
    else:
        raise ValueError("Enter correct DF_FORMAT parameter: 'csv', 'parquet', 'xlsx'.")


    if 'target' in df.columns:
        df = df.rename(columns={'target': 'label'})

    y = df['label']
    X = df.drop(columns=['label'])

    rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=42)

    train_idx, test_idx = next(rskf.split(X, y))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    return X_train, y_train, X_test, y_test


def get_openml_df(name, config):
    
    type_df = config['openml']['type']

    if type_df == 'dataset':
        openml_dfs = config['openml']['dataset']
        dataset = openml.datasets.get_dataset(openml_dfs[name])
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=42)

        train_idx, test_idx = next(rskf.split(X, y))

    elif type_df == 'task':
        openml_tasks = config['openml']['task']
        task = openml.tasks.get_task(openml_tasks[name])
        dataset = task.get_dataset()

        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=task.target_name, dataset_format="dataframe")

        train_idx, test_idx = task.get_train_test_split_indices(fold=0, repeat=0)
        
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    

    if name in ['credit']:
        y_train = y_train.map({'good': 1, 'bad': 0})
        y_test = y_test.map({'good': 1, 'bad': 0})
        
    elif name in ['transfusion', 'fitness', 'diabetes', 'biodegr', 'marketing']:
        y_train = y_train.map({'Yes': 1, 'No': 0})
        y_test = y_test.map({'Yes': 1, 'No': 0})

    if name in ['steel']:

        mapping = {'V1': 'X_Minimum',
                   'V2': 'X_Maximum',
                   'V3': 'Y_Minimum',
                   'V4': 'Y_Maximum',
                   'V5': 'Pixels_Areas',
                   'V6': 'X_Perimeter',
                   'V7': 'Y_Perimeter',
                   'V8': 'Sum_of_Luminosity',
                   'V9': 'Minimum_of_Luminosity',
                   'V10': 'Maximum_of_Luminosity',
                   'V11': 'Length_of_Conveyer',
                   'V12': 'TypeOfSteel_A300',
                   'V13': 'TypeOfSteel_A400',
                   'V14': 'Steel_Plate_Thickness',
                   'V15': 'Edges_Index',
                   'V16': 'Empty_Index',
                   'V17': 'Square_Index',
                   'V18': 'Outside_X_Index',
                   'V19': 'Edges_X_Index',
                   'V20': 'Edges_Y_Index',
                   'V21': 'Outside_Global_Index',
                   'V22': 'LogOfAreas',
                   'V23': 'Log_X_Index',
                   'V24': 'Log_Y_Index',
                   'V25': 'Orientation_Index',
                   'V26': 'Luminosity_Index',
                   'V27': 'SigmoidOfAreas',
                   'V28': 'Pastry',
                   'V29': 'Z_Scratch',
                   'V30': 'K_Scatch',
                   'V31': 'Stains',
                   'V32': 'Dirtiness',
                   'V33': 'Bumps'}

        X_train = X_train.replace(columns=mapping)
        X_test = X_test.replace(columns=mapping)
        
        y_train = y_train.map({'1': 1, '2': 0})
        y_test = y_test.map({'1': 1, '2': 0})

    elif name in ['cancer']:

        y_train = y_train.map({'malignant': 1, 'benign': 0})
        y_test = y_test.map({'malignant': 1, 'benign': 0})

    elif name in ['tae', 'dmft']:

        y_train = y_train.map({'P': 1, 'N': 0})
        y_test = y_test.map({'P': 1, 'N': 0})

    elif name in ['telco']:

        y_train = y_train.map({'Yes': 1, 'No': 0})
        y_test = y_test.map({'Yes': 1, 'No': 0})

    elif name in ['spambase', 'compas', 'crime', 'fraud']:

        y_train = y_train.map({'1': 1, '0': 0})
        y_test = y_test.map({'1': 1, '0': 0})

    elif name in ['pc4', 'kc1']:

        y_train = y_train.map({True: 1, False: 0})
        y_test = y_test.map({True: 1, False: 0})

    elif name in ['irish']:

        y_train = y_train.map({'taken': 1, 'not_taken': 0})
        y_test = y_test.map({'taken': 1, 'not_taken': 0})

    elif name in ['vote']:

        y_train = y_train.map({'democrat': 1, 'republican': 0})
        y_test = y_test.map({'democrat': 1, 'republican': 0})

    elif name in ['cancer']:

        y_train = y_train.map({'malignant': 1, 'benign': 0})
        y_test = y_test.map({'malignant': 1, 'benign': 0})

    elif name in ['creditcard']:

        y_train = y_train.map({'+': 1, '-': 0})
        y_test = y_test.map({'+': 1, '-': 0})

    y_train = y_train.rename('label')
    y_test = y_test.rename('label')
        

    return X_train, y_train, X_test, y_test



def create_cycle_pattern_uneven(sample_0, sample_1, start_with_zero=True):

    zeros_list = sample_0.to_dict('records')
    ones_list = sample_1.to_dict('records')
    
    result_rows = []
    
    if start_with_zero:
        # 0, 1, 0, 1
        max_len = max(len(zeros_list), len(ones_list))
        for i in range(max_len):
            if i < len(zeros_list):
                result_rows.append(zeros_list[i])
            if i < len(ones_list):
                result_rows.append(ones_list[i])
    else:
        # 1, 0, 1, 0
        max_len = max(len(zeros_list), len(ones_list))
        for i in range(max_len):
            if i < len(ones_list):
                result_rows.append(ones_list[i])
            if i < len(zeros_list):
                result_rows.append(zeros_list[i])
    
    if result_rows:
        return pd.DataFrame(result_rows)
    else:
        return pd.DataFrame(columns=sample_0.columns)



def sample_with_ratio(train, n_shots=4, ratio='50/50', regime='default', random_state=42):

    df_train = train.copy().reset_index(drop=True)
    
    df_train['label'] = df_train['label'].astype(int).astype(str)
    
    train_0 = df_train[df_train['label'] == '0']
    train_1 = df_train[df_train['label'] == '1']
    
    ratio_parts = ratio.split('/')
    need_0 = int(int(n_shots) * float(ratio_parts[0]) / 100)
    need_1 = int(int(n_shots) * float(ratio_parts[1]) / 100)
    
    available_0 = len(train_0)
    available_1 = len(train_1)
    
    take_0 = min(need_0, available_0)
    take_1 = min(need_1, available_1)
    
    if take_0 < need_0:
        print(f"ВНИМАНИЕ: Запрошено {need_0} образцов класса '0', но доступно только {available_0}. Берем {take_0}.")
    
    if take_1 < need_1:
        print(f"ВНИМАНИЕ: Запрошено {need_1} образцов класса '1', но доступно только {available_1}. Берем {take_1}.")
    
    total_take = take_0 + take_1
    if total_take < n_shots:
        print(f"ВНИМАНИЕ: Можно взять только {total_take} из {n_shots} запрошенных образцов")
    
    if take_0 > 0:
        sample_0 = train_0.sample(n=take_0, random_state=random_state)
    else:
        sample_0 = pd.DataFrame(columns=df_train.columns)
    
    if take_1 > 0:
        sample_1 = train_1.sample(n=take_1, random_state=random_state)
    else:
        sample_1 = pd.DataFrame(columns=df_train.columns)
    
    if regime == 'halves: zeros_first':  # 50/50 : (0,0,1,1)
        if len(sample_0) > 0 and len(sample_1) > 0:
            result = pd.concat([sample_0, sample_1])
        elif len(sample_0) > 0:
            result = sample_0
        elif len(sample_1) > 0:
            result = sample_1
        else:
            raise ValueError("Нет данных для создания выборки")

    elif regime == 'halves: ones_first':  # 50/50 : (1,1,0,0)
        if len(sample_0) > 0 and len(sample_1) > 0:
            result = pd.concat([sample_1, sample_0])
        elif len(sample_1) > 0:
            result = sample_1
        elif len(sample_0) > 0:
            result = sample_0
        else:
            raise ValueError("Нет данных для создания выборки")
    
    elif regime == 'cycle: zeros_first':  # 50/50: (0,1,0,1)
        if len(sample_0) > 0 and len(sample_1) > 0:
            result = create_cycle_pattern_uneven(sample_0, sample_1, start_with_zero=True)
        else:
            result = pd.concat([sample_0, sample_1]).reset_index(drop=True)

    elif regime == 'cycle: ones_first':  # 50/50: (1,0,1,0)
        if len(sample_0) > 0 and len(sample_1) > 0:
            result = create_cycle_pattern_uneven(sample_0, sample_1, start_with_zero=False)
        else:
            result = pd.concat([sample_0, sample_1]).reset_index(drop=True)

    elif regime == 'default': 
        result = pd.concat([sample_0, sample_1])
        if len(result) > 0:
            result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)

    elif regime == 'full: zeros':
        if take_0 > 0:
            result = sample_0
        else:
            raise ValueError("Нет образцов класса '0' для режима 'full: zeros'")

    elif regime == 'full: ones':
        if take_1 > 0:
            result = sample_1
        else:
            raise ValueError("Нет образцов класса '1' для режима 'full: ones'")
    
    else:
        raise ValueError("Incorrect regime type: enter either default, full, cycle or halves")
    
    if len(result) > 0:
        result = result.reset_index(drop=True)
    
    print(f"Итоговый размер выборки: {len(result)} образцов "
          f"(класс '0': {(result['label'] == '0').sum()}, "
          f"класс '1': {(result['label'] == '1').sum()})")
    
    return result



def save_results(df_final, df_name, df_path):

    for key, df in df_final.items():
        filename = f"{df_name}_{key}.csv"
        df.to_csv(df_path+filename, index=True)
        print(f"Saved {filename}")



def stratified_sample(df, target_col='target', n_samples=1000, random_state=42):

    if n_samples > len(df):
        print(f"Предупреждение: n_samples ({n_samples}) > размер датасета ({len(df)})")
        return df.copy()
    
    class_proportions = df[target_col].value_counts(normalize=True)
    
    samples = []
    for class_value, proportion in class_proportions.items():
        n_class_samples = int(n_samples * proportion)
        class_data = df[df[target_col] == class_value]
        
        if n_class_samples >= len(class_data):
            samples.append(class_data)
        else:
            samples.append(class_data.sample(n=n_class_samples, random_state=random_state))
    
    result = pd.concat(samples)

    if len(result) != n_samples:
        difference = n_samples - len(result)
        if difference > 0:

            remaining = df.drop(result.index)
            additional = remaining.sample(n=difference, random_state=random_state)
            result = pd.concat([result, additional])
        else:
            result = result.sample(n=n_samples, random_state=random_state)
    
    return result.reset_index(drop=True)





def aggregate_results(data_dict, df_name, df_path):
    
    try:
        combined = pd.concat(
            [df.set_index('Unnamed: 0') for df in data_dict.values()],
            keys=data_dict.keys(),
            axis=0
        )
        
    except:
        
        combined = pd.concat(
            [df for df in data_dict.values()],
            keys=data_dict.keys(),
            axis=0
        )

    numeric_cols = combined.select_dtypes(include=['number']).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found for aggregation")
    
    expected_metrics = {'roc_auc', 'f1'}
    available_metrics = set(numeric_cols) & expected_metrics
    if not available_metrics:
        raise ValueError(f"None of the expected metrics {expected_metrics} found in numeric columns")
    
    metrics = combined[list(available_metrics)]
    
    try:
        aggregated_mean = metrics.groupby(level=1).mean()
        aggregated_std = metrics.groupby(level=1).std()
    except TypeError as e:
        raise TypeError(f"Aggregation failed - please check all metric columns are numeric. Original error: {str(e)}")
    
    aggregated_df = pd.concat(
        [aggregated_mean, aggregated_std],
        axis=1,
        keys=['mean', 'std']
    )
    
    aggregated_df.columns = [
        f"{metric}_{stat}" 
        for stat, metric in aggregated_df.columns
    ]
    
    column_order = []
    for metric in available_metrics:
        column_order.extend([f"{metric}_mean", f"{metric}_std"])

    aggregated_df[column_order].to_csv(df_path+df_name+'_agr.csv', index=True)

    #df.index = ['1_old', '2_old', '3_old', 'json_new', 'datamatrix_new', 'latex_new', 'csv_new', 'html_new', 'markdown_new']
    
    return aggregated_df[column_order]




def aggregate_metrics(data_dict):

    try:
        combined = pd.concat(
            [df.set_index('Unnamed: 0') for df in data_dict.values()],
            keys=data_dict.keys(),
            axis=0
        )
    except:
        combined = pd.concat(
            [df for df in data_dict.values()],
            keys=data_dict.keys(),
            axis=0
        )
        
    numeric_cols = combined.select_dtypes(include=['number']).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found for aggregation")
    
    expected_metrics = {'roc_auc', 'f1', 'time'}
    available_metrics = set(numeric_cols) & expected_metrics
    if not available_metrics:
        raise ValueError(f"None of the expected metrics {expected_metrics} found in numeric columns")
    
    metrics = combined[list(available_metrics)]
    
    try:
        aggregated_mean = metrics.groupby(level=1).mean()
        aggregated_std = metrics.groupby(level=1).std()
    except TypeError as e:
        raise TypeError(f"Aggregation failed - please check all metric columns are numeric. Original error: {str(e)}")
    
    aggregated_df = pd.concat(
        [aggregated_mean, aggregated_std],
        axis=1,
        keys=['mean', 'std']
    )
    
    aggregated_df.columns = [
        f"{metric}_{stat}" 
        for stat, metric in aggregated_df.columns
    ]

    column_order = []
    for metric in available_metrics:
        column_order.extend([f"{metric}_mean", f"{metric}_std"])
    
    return aggregated_df[column_order]




def calculate_metrics(true_labels, pred_labels, pred_probs):

    true_labels = [int(elem) for elem in true_labels]
    pred_labels = [int(elem) for elem in pred_labels]
    
    roc_auc = roc_auc_score(true_labels, pred_probs)
    f1 = f1_score(true_labels, pred_labels)

    if roc_auc < 0.5:

        print(f"ROC-AUC: {1 - roc_auc:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return 1-roc_auc, f1
        
    else:
        
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        return roc_auc, f1



def find_last_target_tokens(tokens, model_family):

    if model_family == 'google':
        matches = [(i, tok) for i, tok in enumerate(tokens) if tok in {235276, 235274}]
    else:
        matches = [(i, tok) for i, tok in enumerate(tokens) if tok in {15, 16}]
        
    if not matches:
        last_pos, last_token = (None, None)
    else:
        last_pos, last_token = matches[-1]
    
    return (last_token, last_pos)




def number_to_words(number):

    p = inflect.engine()
    try:
        if isinstance(number, (int, float)) and not pd.isna(number):
            if number == int(number):
                return p.number_to_words(int(number))
            else:
                return p.number_to_words(number)
        else:
            return number
    except:
        return number 


def dataframe_numbers_to_words(df):

    return df.applymap(number_to_words)



def min_max_scale_features(df, feature_range=(0, 1), exclude_columns=None):

    if exclude_columns is None:
        exclude_columns = []
    
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = [col for col in numerical_cols if col not in exclude_columns]
    
    if not numerical_cols:
        return df
    
    scaler = MinMaxScaler(feature_range=feature_range)
    
    df_scaled = df.copy()
    df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
    
    return df_scaled


def transform_dataset(df, config):

    df_transformer_regime = config['data']['DF_TRANSFORMATION_REGIME']
    data_name = config['data']['DATASET_NAME']

    if df_transformer_regime == 'scaling':
        return min_max_scale_features(df)
    elif df_transformer_regime == 'words':
        return dataframe_numbers_to_words(df)
    elif df_transformer_regime == 'shuffling':
        
        cols = df.columns.tolist()
        print(cols)
        np.random.seed(42)
        np.random.shuffle(cols)
        print(cols)
        df = df[cols]
        df = df[cols]
        return df
        
    elif df_transformer_regime == 'change_units' and data_name == 'transfusion':
        
        df['YearsSinceLastDonation'] = df['MonthsSinceLastDonation']/12
        df['YearsSinceFirstDonation'] = df['MonthsSinceFirstDonation']/12
        df['TotalBloodDonated'] = df['TotalBloodDonated']/1000

        df = df.drop(columns = ['MonthsSinceLastDonation'])
        df = df.drop(columns = ['MonthsSinceFirstDonation'])

        return df
        
    elif df_transformer_regime == 'change_units' and data_name == 'credit':

        df['credit_duration_years'] = df['duration_months']/12
        df['installment_rate_share'] = df['installment_rate_percent']/100
        df['credit_amount_thousand_DM'] = df['credit_amount']/1000
        df['residence_duration_months'] = df['residence_since']*12
        df['age_months'] = df['age_years'].apply(lambda x: int(x)*12)
        df['number_of_existing_credits'] =df['existing_credits_count']

        df = df.drop(columns = ['duration_months', 'installment_rate_percent', 'credit_amount', 'residence_since', 'existing_credits_count', 'age_years'])

        return df
        
    else:
        return df



# Revise on code review: the logic of train-test split, is it coherent with TabArena precisely?

def process_syntethic_df(params):

    dataset_path = params['DATASET_PATH_IF_SYNT']+f"{params['DATASET_NAME']}.csv"
    df = pd.read_csv(dataset_path, index_col=[0])
    df = df.loc[:, ~df.columns.isin(['Unnamed: 0'])]
    df = df.astype(str)

    # TODO: снести в предобработку самого датасета на этапе генерации, а неx в препроцессинг
    
    if 'salary' in df.columns.tolist():
        df['salary'] = df['salary'].astype(float).round()

    column_renames = {
        'likes_coffee': 'label',
        'target': 'label',
        'Loan_Status': 'label'
    }
    df = df.rename(columns={k: v for k, v in column_renames.items() if k in df.columns})

    y = df['label'].astype(int).astype(str)
    X = df.drop('label', axis=1)

    rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=42)

    train_idx, test_idx = next(rskf.split(X, y))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    return X_train, y_train, X_test, y_test 




def save_probs(pred_probs, true_labels, pred_labels, config, current_serialization, current_rs):

    local_llm = config['experiment']['local_llm']
    dataset_name = config['data']['DATASET_NAME']
    config_code = config['experiment']['CONFIG_CODE']
    n_shot = config['experiment']['N_SHOTS']
    regimme = config['experiment']['regime']

    if local_llm:
        model_name = config['local_model']['name'].split('/')[-1].lower()
    else:  
        # TODO: revise on code review
        pass

    
    dir_path = os.path.join(
        config['data']['PROBS_PATH'],
        f"{config['data']['DF_TYPE']}_datasets",
        dataset_name,
        f"{n_shot}_shots",
        model_name,
        f"rs_{regimme}"
    )
    

    os.makedirs(dir_path, exist_ok=True)
    filename = f"df_{n_shot}fs_{model_name}_{regimme}_{dataset_name}_{current_serialization}_{current_rs}_{config_code}.pkl"
    
    full_path = os.path.join(dir_path, filename)
    

    results = {
        'pred_probs': pred_probs,
        'true_labels': true_labels,
        'pred_labels': pred_labels,
        'timestamp': datetime.now().isoformat(),
        'params': config,
        'serialization': current_serialization,
        'random_state': current_rs
    }

    try:
        with open(full_path, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Сохранено: {full_path}')
        
    except Exception as e:
        print(f'Ошибка при сохранении {full_path}: {e}')
