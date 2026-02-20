
import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tiktoken

from source.prompts import (
    system_prompt
)
from source.models import (
    PromptFactory
)

METRIC_KEYS = [
    "total_tokens", "unique_tokens", "numeric_tokens",
    "alpha_tokens", "punctuation_tokens", "whitespace_tokens", "char_count",
    "chars_per_token", "repetition_rate", "delimiter_count", "delimiter_token_ratio", "important_tokens"
]

DEBUG_LIMIT = None

class DatasetMetaFeatures:
    def __init__(self, df: pd.DataFrame, dataset_name: str):
        self.df = df
        self.dataset_name = dataset_name
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
        self.meta_features = {}

        self.meta_features['dataset'] = self.dataset_name

    def calculate_all(self):

        # Total features
        self.meta_features['total_features'] = len(self.df.columns)

        # Categorical features
        cat_count = 0
        for col in self.categorical_cols:
            if self.df[col].nunique() < 50:
                cat_count += 1
        self.meta_features['categorical_features'] = cat_count

        # Boolean features
        bool_count = 0
        for col in self.df.columns:
            unique_vals = set(self.df[col].dropna().unique())
            if unique_vals.issubset({0, 1}) or unique_vals.issubset({True, False}):
                bool_count += 1
        self.meta_features['boolean_features'] = bool_count

        # Text features
        text_count = 0
        for col in self.categorical_cols:
            if self.df[col].nunique() >= 50:
                text_count += 1
            elif self.df[col].astype(str).str.len().mean() > 50:
                text_count += 1
        self.meta_features['text_features'] = text_count

        # Features with units/symbols
        unit_pattern = re.compile(r'[\$€£¥%kggmlbftin°C]', re.IGNORECASE)
        unit_cols = 0
        for col in self.df.columns:
            sample = self.df[col].astype(str).sample(min(100, len(self.df))).dropna()
            if sample.str.contains(unit_pattern).any():
                unit_cols += 1
        self.meta_features['features_with_units'] = unit_cols

        # Date features
        date_count = 0
        date_cols = self.df.select_dtypes(include=['datetime64', 'datetimetz']).columns
        date_count += len(date_cols)
        date_pattern = re.compile(r'\d{4}[-/]\d{2}[-/]\d{2}')
        for col in self.categorical_cols:
            if col in date_cols: continue
            sample = self.df[col].astype(str).sample(min(100, len(self.df))).dropna()
            if sample.str.contains(date_pattern).any():
                date_count += 1
        self.meta_features['date_features_count'] = date_count


        # Max length of categorical feature
        max_cat_len = 0
        for col in self.categorical_cols:
            if self.df[col].dtype == 'object':
                col_max_len = self.df[col].astype(str).str.len().max()
                if col_max_len > max_cat_len:
                    max_cat_len = col_max_len
        self.meta_features['max_categorical_feature_length'] = max_cat_len

        int_cols = self.df.select_dtypes(include=['int64', 'int32']).columns
        float_cols = self.df.select_dtypes(include=['float64', 'float32']).columns

        total_numeric = len(self.numeric_cols)
        if total_numeric > 0:
            self.meta_features['ratio_float_to_numeric'] = len(float_cols) / total_numeric
            self.meta_features['ratio_int_to_numeric'] = len(int_cols) / total_numeric

            decimal_places_sum = 0
            float_count = 0
            for col in float_cols:
                series = self.df[col].dropna().abs()
                if len(series) == 0:
                    continue

                n_samples = min(50, len(series))
                sample = series.sample(n_samples)

                for val in sample:
                    s_val = f"{val:.10f}".rstrip('0').rstrip('.')
                    if '.' in s_val:
                        decimals = len(s_val.split('.')[1])
                        decimal_places_sum += decimals
                        float_count += 1

            avg_decimals = (decimal_places_sum / float_count) if float_count > 0 else 0
            self.meta_features['avg_decimal_precision_in_floats'] = avg_decimals
        else:
            self.meta_features['ratio_float_to_numeric'] = 0
            self.meta_features['avg_decimal_precision_in_floats'] = 0

        # Spread of numeric features (Max - Min)
        max_spread = 0
        for col in self.numeric_cols:
            spread = self.df[col].max() - self.df[col].min()
            if spread > max_spread:
                max_spread = spread
        self.meta_features['max_numeric_spread'] = max_spread

        # Missing values count
        self.meta_features['total_missing_values'] = int(self.df.isnull().sum().sum())
        self.meta_features['missing_values_ratio'] = float(self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1]))

        # Abbreviations count
        abbr_pattern = re.compile(r'\b[A-Z]{2,5}\b')
        total_abbrs = 0
        for col in self.df.columns:
            matches = re.findall(abbr_pattern, col)
            total_abbrs += len(matches)
        for col in self.df.columns:
            sample = self.df[col].astype(str).sample(min(20, len(self.df))).dropna()
            for text in sample:
                matches = re.findall(abbr_pattern, text)
                total_abbrs += len(matches)
        self.meta_features['abbreviation_count'] = total_abbrs


        return self.meta_features




def get_prompt_metadata(messages, model_family, tokenizer=None):

    if model_family == 'openai':
        enc = tiktoken.encoding_for_model("gpt-4o-mini")
        text = openai_chat_template(messages)
        tokens_ids = enc.encode(text)
        tokens_str = enc.decode_tokens_bytes(tokens_ids)

        text_after_predict = text.split("**PREDICT**", 1)[1]
        important = enc.encode(text_after_predict)
        important_tokens = len(important)

    else:
        text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False)

        tokens_ids = tokenizer.encode(text, add_special_tokens=True)
        tokens_str = tokenizer.convert_ids_to_tokens(tokens_ids)

        text_after_predict = text.split("**PREDICT**", 1)[1]
        important = tokenizer.encode(text_after_predict, add_special_tokens=True)
        important_tokens = len(important)

    # Basic Token Stats
    total_tokens = len(tokens_ids)
    unique_tokens = len(set(tokens_ids))
    #special_tokens_map = tokenizer.special_tokens_map.values()

    # Content Analysis (Character-level on decoded tokens)
    numeric_tokens = 0
    alpha_tokens = 0
    punctuation_tokens = 0
    whitespace_tokens = 0

    for t in tokens_str:
        if model_family == 'qwen':
            clean_t = t.replace('Ġ', ' ').replace('▁', ' ')
        else:
            clean_t = t.decode('utf-8')

        if clean_t.strip() == '':
            whitespace_tokens += 1
        elif clean_t.replace('.','',1).replace('-','',1).isdigit():
            numeric_tokens += 1
        elif clean_t.isalpha():
            alpha_tokens += 1
        else:
            punctuation_tokens += 1

    # Efficiency Metrics
    char_count = len(text)
    chars_per_token = char_count / total_tokens if total_tokens > 0 else 0
    repetition_rate = 1 - (unique_tokens / total_tokens) if total_tokens > 0 else 0

    # Tabular/Structure Heuristics
    delimiter_chars = [',', '|', '\t', ';']
    delimiter_count = sum(text.count(d) for d in delimiter_chars)
    delimiter_token_ratio = delimiter_count / total_tokens if total_tokens > 0 else 0


    return (total_tokens,
            unique_tokens,
            numeric_tokens,
            alpha_tokens,
            punctuation_tokens,
            whitespace_tokens,
            char_count,
            chars_per_token,
            repetition_rate,
            delimiter_count,
            delimiter_token_ratio,
            important_tokens
            )


def compute_metrics_for_dataset(serialization_test, config, schema, system_prompt, task_description, few_shot_examples, model_family, MODEL_NAME, tokenizer=None):
    metrics_accumulator = {key: [] for key in METRIC_KEYS}

    limit = len(serialization_test) if DEBUG_LIMIT is None else min(DEBUG_LIMIT, len(serialization_test))

    for i in tqdm(range(limit), desc="Samples", leave=False):
        messages = PromptFactory.create_prompt(
            config=config, schema=schema, system_prompt=system_prompt,
            task_description=task_description, serialization_train=few_shot_examples,
            predict_string=str(serialization_test[i]), local_llm=True
        )

        if MODEL_NAME == "openai/gpt-4o-mini":
            results = get_prompt_metadata(messages, model_family)
        else:
            results = get_prompt_metadata(messages, model_family, tokenizer)
        for key, value in zip(METRIC_KEYS, results):
            metrics_accumulator[key].append(value)

    return {key: np.mean(values) if values else 0.0 for key, values in metrics_accumulator.items()}


def openai_chat_template(messages):
    chat = ""
    for msg in messages:
        chat += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    chat += "<|im_start|>assistant\n"
    return chat