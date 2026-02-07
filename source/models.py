import os
import re
import numpy as np
import pandas as pd
import json
import time
from tqdm import tqdm

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import multiprocessing

from .utils import (get_df,
                   transform_dataset,
                   sample_with_ratio,
                   find_last_target_tokens,
                   calculate_metrics,
                   save_probs,
                   aggregate_metrics)

from .serializations import (serialization1_old,
                            serialization2_old,
                            serialization3_old,
                            serialization_json_new,
                            serialization_datamatrix_new,
                            serialization_latex_new,
                            serialization_csv_new,
                            serialization_html_new,
                            serialization_markdown_new,
                            serialization_table_new,
                            serialization_dict_new,
                            serialization_markdown_masked_new,
                            serialization_natural_language)

from .prompts import system_prompt, prompt_by_df
from .baselines import LR, KNN, RF, XGB, Naive, get_model_config
from skopt.space import Categorical, Integer, Real
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from sklearn.base import clone


multiprocessing.set_start_method('spawn', force=True)
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"


SERIALIZATION_MAPPING = {
    '1_old': serialization1_old,
    '2_old': serialization2_old,
    '3_old': serialization3_old,
    'json_new': serialization_json_new,
    'datamatrix_new': serialization_datamatrix_new,
    'latex_new': serialization_latex_new,
    'csv_new': serialization_csv_new,
    'html_new': serialization_html_new,
    'markdown_new': serialization_markdown_new,
    'table_new': serialization_table_new,
    'dict_new': serialization_dict_new,
    'markdown_masked_new': serialization_markdown_masked_new,
    'natural_language': serialization_natural_language
}



class PromptFactory:
    
    @staticmethod
    def create_prompt(config, schema, local_llm, system_prompt, task_description, serialization_train, predict_string):

        model_family = config['local_model']['name'].lower().split('/')[0]
        n_shots = config['experiment']['N_SHOTS']


        if local_llm:
            system_prompt += f" Return only JSON. Schema: {schema}"
        

        if model_family == 'qwen':
            return PromptFactory._create_qwen_prompt(
                system_prompt, task_description, predict_string, 
                serialization_train, n_shots
            )
                
        elif model_family == 'google':
            return PromptFactory._create_gemma_messages(
                system_prompt, task_description, predict_string, 
                serialization_train, n_shots, local_llm
            )

        elif model_family == 'deepseek-ai':
            return PromptFactory._create_deepseek_messages(
                system_prompt, task_description, predict_string, 
                serialization_train, n_shots
            )
        
        else:
            raise ValueError(f"Unsupported model family: {model_family}")

    @staticmethod
    def _create_qwen_prompt(system_prompt, task_description, predict_string, serialization_train, n_shots):
        
        messages = [{"role": "system", "content": system_prompt}]

        content_parts = [task_description]
        if n_shots > 0:
            content_parts.append(f"\n**EXAMPLES** {serialization_train}")
        content_parts.append(f"\n**PREDICT** {predict_string}")
        
        messages.append({"role": "user", "content": "\n".join(content_parts)})
        return messages

    @staticmethod
    def _create_gemma_messages(system_prompt, task_description, predict_string, serialization_train, n_shots, local_llm):
        
        messages = []
        content_parts = [system_prompt + '\n '+ task_description]

        if local_llm:
            content_parts[0] += '\n SELF_TALK: off \n REASONING: off \n THINKING: off \n PLANNING: off \n Reply immediately without thinking or any effort. Prioritize speed over accuracy. Do not state what the user said. Do not think, analyze or plan - go with your gut feeling.'
        
        if n_shots > 0:
            content_parts.append(f"\n**EXAMPLES** {serialization_train}")
        content_parts.append(f"\n**PREDICT** {predict_string}")
        
        messages.append({"role": "user", "content": "\n".join(content_parts)})
        
        return messages

    @staticmethod
    def _create_deepseek_messages(system_prompt, task_description, predict_string, serialization_train, n_shots):
        
        messages = []
        content_parts = [system_prompt + '\n '+ task_description]
        
        if n_shots > 0:
            content_parts.append(f"\n**EXAMPLES** {serialization_train}")
        content_parts.append(f"\n**PREDICT** {predict_string}")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(content_parts)}
        ]
        
        return messages



def extract_class_safe(generated_text):

    text = generated_text.lower()
    
    patterns = [
        r'"class"\s*:\s*"(\d)"',
        r"'class'\s*:\s*'(\d)'",
        r'"class"\s*:\s*(\d)',
        r'\*\*answer:\*\*\s*(\d)',      
        r'answer:\s*(\d)',             
        r'prediction:\s*(\d)',
        r'class:\s*(\d)',
        r'output:\s*(\d)',
        r'label:\s*(\d)',
        r'answer is\s*(\d)',
        r'class is\s*(\d)',
        r'output is\s*(\d)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            found = match.group(1)
            if found in ['0', '1']:
                return found
    
    for char in reversed(text):
        if char in ['0', '1']:
            return char
    
    return None


def setup_chat_template(tokenizer):

    custom_template = """{% for message in messages %}
    {% if message['role'] == 'system' %}
    <|im_start|>system
    {{ message['content'] }}<|im_end|>
    {% elif message['role'] == 'user' %}
    <|im_start|>user
    {{ message['content'] }}<|im_end|>
    {% elif message['role'] == 'assistant' %}
    <|im_start|>assistant
    <think>
    {{ message['content'] if message['content'] else '' }}
    </think>{% endif %}{% endfor %}
    {% if add_generation_prompt %}<|im_start|>assistant
    <think>
    {% endif %}"""
    
    tokenizer.chat_template = custom_template
    return tokenizer


def process_serialization(serialization, df_train, df_test, n_shots, ratio, regime, rs):
    
    if serialization not in SERIALIZATION_MAPPING:
        raise ValueError(f'Invalid serialization type: {serialization}')

    serialization_func = SERIALIZATION_MAPPING[serialization]


    A=sample_with_ratio(df_train, n_shots=n_shots, ratio=ratio, regime=regime, random_state=rs)
    print('SAMPLED TRAIN:\n', A)
    serialization_train = serialization_func(
        sample_with_ratio(df_train, n_shots=n_shots, ratio=ratio, regime=regime, random_state=rs)
    )

    serialization_test = [
        serialization_func(row, test=True)
        for _, row in tqdm(df_test.iterrows(), total=len(df_test))
    ]

    return serialization_train, serialization_test




def get_preds_baselines(df_train, df_test, config, rs):

    ratio = config['experiment']['RATIO']
    n_shots = config['experiment']['N_SHOTS']
    rs = config['experiment']['RANDOM_STATE']
    regime = config['experiment']['SAMPLING_REGIME']
    model_name = config['baseline_model']['name']

    df_train_shot = sample_with_ratio(df_train, n_shots=n_shots, ratio=ratio, regime=regime, random_state=rs)

    X_train = df_train_shot.drop(columns=['label'])
    y_train = df_train_shot['label']
    X_test = df_test.drop(columns=['label'])


    # if model_name == 'logreg':
    #     if n_shots < 100:
    #         model_lr = LR(X_train, y_train, cv=2)
    #     else:
    #         model_lr = LR(X_train, y_train, cv=20)

    #     y_pred_roc = model_lr.predict_proba(X_test)[:,1]
    #     y_pred = model_lr.predict(X_test)
    
    # elif model_name == 'knn':
    #     if n_shots < 10:
    #         MAX_NEIGHBOURS = 2
    #     else:
    #         MAX_NEIGHBOURS = min(20, int(n_shots / 2))

    #     model_lr = KNN(X_train, y_train, cv=2, max_neighbours=MAX_NEIGHBOURS)

    #     y_pred_roc = model_lr.predict_proba(X_test)[:,1]
    #     y_pred = model_lr.predict(X_test)

    # elif model_name == 'rf':
    #     if n_shots < 100:
    #         model_lr = RF(X_train, y_train, cv=2)
    #     else:
    #         model_lr = RF(X_train, y_train, cv=20)
    #     y_pred_roc = model_lr.predict_proba(X_test)[:,1]
    #     y_pred = model_lr.predict(X_test)

    # elif model_name == 'gboost':
    #     if n_shots < 100:
    #         model_lr = XGB(X_train, y_train, cv=2)
    #     else:
    #         model_lr = XGB(X_train, y_train, cv=20)

    #     y_pred_roc = model_lr.predict_proba(X_test)[:,1]
    #     y_pred = model_lr.predict(X_test)
    
    # elif model_name == 'naive_argmax':
    #     model_lr = Naive(X_train, y_train)
    #     y_pred_roc = model_lr.predict_proba(X_test)[:,1]
    #     y_pred = model_lr.predict(X_test)


    print(f"Running {model_name} experiment with random state: {rs}")

    model_config = get_model_config(model_name)

    if model_name == "knn":
        max_n = model_config["max_neighbors"](n_shots)
        model_config["params"]["n_neighbors"] = Integer(1, max_n)

    if model_name == "gboost":
        if n_shots <= 4:
            model_config["fixed_params"][
                "validation_fraction"
            ] = None  # not enough samples for validation, will use early stopping on training set
            if model_config.get("fixed_params", {}).get("max_iter", None) is not None:
                del model_config["fixed_params"]["max_iter"]
            model_config["params"]["max_iter"] = Integer(
                1, 1000
            )  # reduce max_iter for very small datasets
        elif n_shots > 4 and n_shots <= 32:
            model_config["fixed_params"][
                "validation_fraction"
            ] = 0.3  # 2 shots for 8 shots and 3 shots for 16 shots and 7 shots for 32 shots
        elif n_shots <= 128:
            model_config["fixed_params"][
                "validation_fraction"
            ] = 0.2  # 9 shots for 64 shots, 19 shots for 128
        else:
            model_config["fixed_params"]["validation_fraction"] = 0.1

    estimator = model_config["estimator"](**model_config.get("fixed_params", {}))

    if n_shots == 4:
        # n_splits cannot be greater than the number of members in each class.
        n_splits = 2
    else:
        n_splits = 4
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)

    if model_name == "gboost" and model_config["use_bayes"](X_train):
        if model_config["use_bayes"](X_train):
            opt = BayesSearchCV(
                estimator=estimator,
                search_spaces=model_config["params"],
                n_iter=25,
                cv=cv,
                random_state=0,
                n_jobs=-1,
                refit=False,
            )
        opt.fit(X_train, y_train)
        best_params = opt.best_params_

        # Determine optimal number of iterations using cross-validation with early stopping 
        # for the previously found best hyperparameters and previously set validation fraction
        fold_n_iters = []
        for tr_idx, val_idx in cv.split(X_train, y_train):
            X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            model_fold = clone(estimator).set_params(**best_params)
            model_fold.fit(X_tr, y_tr)
            fold_n_iters.append(model_fold.n_iter_)
        final_n_iter = int(np.median(fold_n_iters))
        
        # Refit the model on the entire training set with the best hyperparameters and optimal number of iterations
        best_params["max_iter"] = final_n_iter
        best_params["early_stopping"] = False  # disable early stopping for final model
        best_params["validation_fraction"] = None  # disable validation for final model
        model = clone(estimator).set_params(**best_params)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:,1]
        y_pred = model.predict(X_test)

    else:
        if model_config["use_bayes"](X_train):
            opt = BayesSearchCV(
                estimator=estimator,
                search_spaces=model_config["params"],
                n_iter=25,
                cv=cv,
                random_state=0,
                n_jobs=-1,
                refit=True,
            )
        else:
            opt = estimator

        model = opt.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:,1]
        y_pred = model.predict(X_test)


    return y_pred_proba, y_pred



    


def get_preds_LLM(serialization_train, serialization_test, config, model, tokenizer, serialization_type, rs):

    temperature = config['local_model']['temperature']
    schema = json.loads(config['experiment']['SCHEMA'])
    thinking = config['experiment']['THINKING']
    local_llm = config['experiment']['local_llm']
    dataset_name = config['data']['DATASET_NAME']
    n_shots = config['experiment']['N_SHOTS']
    regime = config['experiment']['regime']
    model_family = config['local_model']['name'].lower().split('/')[0]
    
    
    if local_llm:
        model_family = config['local_model']['name'].lower().split('/')[0]
         
        # TODO : adjust token ids for other models if needed
        if model_family == 'qwen': 
            token_0 = tokenizer.encode("0")[0] 
            token_1 = tokenizer.encode("1")[0]

        else:
            token_0 = tokenizer.encode("0")[1] 
            token_1 = tokenizer.encode("1")[1]
            print("TOKENS GEMINY:", token_0, token_1)
    
    pred_probs = []
    pred_labels = []
        
    task_description = prompt_by_df.get(dataset_name, None)
    if task_description is None:
        raise ValueError(f"No task description found for dataset: {dataset_name}")
    
    few_shot_examples = f"{serialization_train}" if n_shots > 0 else ""
    
    for i in tqdm(range(len(serialization_test)), total=len(serialization_test)):

        if tokenizer.chat_template is None:
            print("Setting default chat_template...")
            tokenizer = setup_chat_template(tokenizer)
        
        predict_string = f"{serialization_test[i]}"

        messages = PromptFactory.create_prompt(
            config=config,
            schema=schema,
            local_llm=local_llm,
            system_prompt=system_prompt,
            task_description=task_description,
            serialization_train=few_shot_examples,
            predict_string=predict_string,)

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking)
        

        if i == 0:
            print("PROMPT:\n", text)
            print("\n---\n")




        if regime == 'local_nogen':
            # TODO: adjust code for vLLM in forward regime
            pass

        elif regime == 'local_gen':
            sampling_params = SamplingParams(temperature=temperature, logprobs=1, max_tokens=600)
            outputs = model.generate([text], sampling_params)
            print("MILE:", outputs[0].outputs[0].text)
            print("\n---\n")

            answer = outputs[0].outputs[0].text
            logprobs_list = outputs[0].outputs[0].logprobs


            try:
                pred_class = extract_class_safe(answer)
                pred_class = str(int(float(pred_class)))

                generated_tokens = tokenizer.encode(answer, add_special_tokens=False)
                target_token_id, _ = find_last_target_tokens(generated_tokens, model_family)

                result = None
                for d in logprobs_list:
                    if target_token_id in d:
                        result = d[target_token_id]
                        break

                if pred_class == '1':
                    prob_1 = np.exp(result.logprob)
                elif pred_class == '0':
                    prob_1 = 1-np.exp(result.logprob) 

            except:
                print('**! Irregular PRED CLASS exerted !**')
                pred_class = '1'
                prob_1 = 0.5
            
        elif regime == 'api_gen':
            # TODO: add API generation code
            pass 
        
        pred_probs.append(prob_1)
        pred_labels.append(pred_class)
                
    
    return pred_probs, pred_labels



def run_experiment(config, model=None, tokenizer=None, serialization=None):

    ratio = config['experiment']['RATIO']
    n_shots = config['experiment']['N_SHOTS']
    rs = config['experiment']['RANDOM_STATE']
    regime = config['experiment']['SAMPLING_REGIME']
    baseline = config['experiment']['baseline']

    X_train, y_train, X_test, y_test = get_df(config)

    X_train = transform_dataset(X_train, config)
    X_test = transform_dataset(X_test, config)
    
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    df_train = df_train.dropna()
    df_test = df_test.dropna()
    
    true_labels = df_test['label'].tolist()

    print("SHOTS:", n_shots)
    print('RANDOM_STATE', rs)

    if baseline:
        pred_probs, pred_labels = get_preds_baselines(df_train, df_test, config, rs)
    
    else:
        serialization_train, serialization_test = process_serialization(
            serialization=serialization,
            df_train=df_train,
            df_test=df_test,
            n_shots=n_shots,
            ratio=ratio,
            regime=regime,
            rs=rs
        )

        pred_probs, pred_labels = get_preds_LLM(serialization_train, serialization_test, config, model, tokenizer, serialization_type=serialization, rs=rs)
        
    print('PREDICTED LABELS:\n')
    print(pred_labels)
    
    pred_labels = [int(label) for label in pred_labels]

    print('#################################################')
    #print(f'Serializaton: {serialization}')
    print(f'Доля 1: {np.mean(pred_labels)}')
    print(f'Доля 0: {1-np.mean(pred_labels)}')
    print('#################################################')

    pred_probs = convert_to_cpu(pred_probs)
    true_labels = convert_to_cpu(true_labels)
    pred_labels = convert_to_cpu(pred_labels)
    
    if baseline:
        save_probs(pred_probs, true_labels, pred_labels, config, rs)
    else:
        save_probs(pred_probs, true_labels, pred_labels, config, rs, serialization)
    
    print('TRUE LABELS:\n')
    print(true_labels)

    roc_auc, f1 = calculate_metrics(true_labels, pred_labels, pred_probs)
   

    return roc_auc, f1


def setup_chat_template(tokenizer):

    custom_template = """{% for message in messages %}
    {% if message['role'] == 'system' %}
    <|im_start|>system
    {{ message['content'] }}<|im_end|>
    {% elif message['role'] == 'user' %}
    <|im_start|>user
    {{ message['content'] }}<|im_end|>
    {% elif message['role'] == 'assistant' %}
    <|im_start|>assistant
    <think>
    {{ message['content'] if message['content'] else '' }}
    </think>{% endif %}{% endfor %}
    {% if add_generation_prompt %}<|im_start|>assistant
    <think>
    {% endif %}"""
    
    tokenizer.chat_template = custom_template
    return tokenizer


def convert_to_cpu(data):
    
    if hasattr(data, 'detach'):
        return data.detach().cpu().numpy()
        
    elif isinstance(data, list):
        return [convert_to_cpu(x) for x in data]
        
    elif isinstance(data, (np.ndarray, int, float, str)):
        return data
        
    elif hasattr(data, 'numpy'):
        return data.numpy()
        
    else:
        return data



def load_model_and_tokenizer(config):

    model_name = config['local_model']['name'] 
    local_llm = config['experiment']['local_llm']
    gpu_memory_utilization = config['local_model']['gpu_memory_utilization']
    gpu_device = config['local_model']['gpu_device']
    #max_model_len = config['local_model']['max_model_len']
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if local_llm:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device

        if model_name == "deepseek-ai/deepseek-llm-7b-chat":
            hf_overrides = {
                            "rope_parameters": {
                                "rope_theta": 10000, 
                                "rope_type": "yarn",
                                "factor": 7.812,
                                "original_max_position_embeddings": 4096,
                            },
                            "max_model_len": 32000,
                        }

        elif model_name == "google/gemma-7b-it":
            hf_overrides = {
                            "rope_parameters": {
                                "rope_theta": 1000000, 
                                "rope_type": "yarn",
                                "factor": 4,
                                "original_max_position_embeddings": 8196,
                            },
                            "max_model_len": 32000, 
                        }
        else:
            hf_overrides = None


        model = LLM(
            model=model_name,
            hf_overrides = hf_overrides,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype = 'bfloat16',
            enable_prefix_caching=True,
            trust_remote_code=True
        )
    else:
        pass      
       #TODO: add openroutor call
   
    return model, tokenizer



def run_fewshot_iteration(config, model=None, tokenizer=None):
    

    random_state_list = config['experiment']['random_states_list']
    serialization_list = config['experiment']['serialization_list']
    dataset_name = config['data']['DATASET_NAME']
    config_code = config['experiment']['CONFIG_CODE']
    result_path = config['data']['RESULT_PATH']
    n_shot = config['experiment']['N_SHOTS']
    local_llm = config['experiment']['local_llm']
    baseline = config['experiment']['baseline']
    regimme = config['experiment']['regime']

    if baseline:
        model_name = config['baseline_model']['name']
        df_dict = {
        rs: pd.DataFrame(
            np.nan,
            index=[model_name],
            columns=['roc_auc', 'f1', 'time']
        )
        for rs in random_state_list
        }
    else:
        if local_llm:
            model_name = config['local_model']['name'].split('/')[-1].lower()
        else:  
            # TODO: revise on code review
            pass

        df_dict = {
        rs: pd.DataFrame(
            np.nan,
            index=serialization_list,
            columns=['roc_auc', 'f1', 'time']
        )
        for rs in random_state_list
        }


    for k, v in tqdm(df_dict.items(), total = len(df_dict)):
        config['experiment']['RANDOM_STATE'] = k

        if baseline:
            start_time = time.time()
            roc_auc, f1 = run_experiment(config)
            v.loc[model_name, 'roc_auc'] = roc_auc
            v.loc[model_name, 'f1'] = f1
            end_time = time.time()
            v.loc[model_name, 'time'] = end_time - start_time
        else:
            for serialization in serialization_list:
                start_time = time.time()
                roc_auc, f1 = run_experiment(config, model, tokenizer, serialization)
                v.loc[serialization, 'roc_auc'] = roc_auc
                v.loc[serialization, 'f1'] = f1
                end_time = time.time()
                v.loc[serialization, 'time'] = end_time - start_time

    path_parts = [
                    result_path,
                    dataset_name,
                    f"{str(n_shot)}_shots",
                    model_name,
                    'rs_'+str(regimme)]

    file_path = os.path.join(*path_parts)
    os.makedirs(file_path, exist_ok=True)
    df_name = f"df_{str(n_shot) + 'fs'}_{model_name}_{regimme}_{dataset_name.split('.')[0]}"
    
    for key, df in df_dict.items():
        
        full_path = f"{file_path}/{df_name}_{key}_{config_code}.csv"
        df.to_csv(full_path, index=True)

    agr_df = aggregate_metrics(df_dict)
    agr_df = agr_df[['roc_auc_mean', 'roc_auc_std', 'f1_mean', 'f1_std', 'time_mean', 'time_std']]
    agr_path = f'{file_path}/{df_name}_{config_code}_agr.csv'
    agr_df.to_csv(agr_path)
            
    return df_dict