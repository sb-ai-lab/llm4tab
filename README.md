# LLMTabBench

**LLMTabBench** is a benchmarking framework for evaluating Large Language Models (LLMs) on **binary classification of tabular data** in a **few-shot** setting. The benchmark measures how well LLMs can classify samples of tabular datasets when provided with only a small number of labeled examples (shots) as in-context demonstrations or in the zero-shot settings.

The framework supports:
- **Local LLM inference** via [vLLM](https://github.com/vllm-project/vllm) (Qwen, Gemma, DeepSeek model families)
- **API-based LLM inference** via [OpenRouter](https://openrouter.ai/) (GPT)
- **Classical ML baselines** (Logistic Regression, KNN, Random Forest, Gradient Boosting) with Bayesian hyperparameter optimization
- **Multiple data serialization formats** (Markdown, HTML, JSON, CSV, LaTeX, natural language, and more)
- **OpenML datasets** and **custom local datasets (.csv)**
- Automated metric calculation (ROC-AUC, F1) with aggregation across multiple random seeds

---

## Quick Start

### Prerequisites

- Python ≥ 3.12
- [uv](https://github.com/astral-sh/uv) package manager (installed automatically by `make` if missing)
- NVIDIA GPU with CUDA support (for local LLM inference)

### Installation

```bash
# 1. Clone the repository
git clone git@github.com:sb-ai-lab/llm4tab.git
cd llm4tab

# 2. Set up the environment (creates venv + installs all dependencies)
make setup

# 3. Create .env file and add your API keys (if using API models)
cp .env_example .env
# Edit .env and add your OPENROUTER_KEY if needed

# 4. Run the benchmark
make run
```

This will execute `main.py` using the settings defined in `config.yaml`.

---

## Configuration (`config.yaml`)

All experiment settings are controlled via a single `config.yaml` file. Below is a section-by-section reference.

### Data Source — `data`

```yaml
data:
  DF_TYPE: 'openml'
  DF_FORMAT: 'csv'
  DATASET_NAME: 'airbnb'
  DF_TRANSFORMATION_REGIME: 'no'
  LOCAL_DATASET_PATH: 'datasets/'
  RESULT_PATH: 'results/experiments_result/'
  PROBS_PATH: 'results/llm_probs/'
```

| Parameter | Description |
|---|---|
| `DF_TYPE` | `'openml'` — fetch datasets from OpenML; `'custom'` — load CSVs from `datasets/` folder |
| `DF_FORMAT` | File format for custom datasets (currently only `'csv'` is supported) |
| `DATASET_NAME` | Name of the dataset (auto-set during iteration) |
| `DF_TRANSFORMATION_REGIME` | Optional preprocessing: `'no'` (none), `'scaling'` (MinMax), `'words'` (numbers→words), `'shuffling'` (shuffle columns), `'change_units'` (dataset-specific unit conversion) |
| `LOCAL_DATASET_PATH` | Directory where custom CSV files are stored |
| `RESULT_PATH` | Directory to save experiment result CSVs |
| `PROBS_PATH` | Directory to save prediction probability pickle files |

---

### Running with OpenML Data

Set `DF_TYPE: 'openml'` in the `data` section and configure the `openml` section:

```yaml
data:
  DF_TYPE: 'openml'

openml:
  type: 'dataset'
  df_openml: ['compas', 'vote', 'kc1', 'irish']
  dataset: {
    telco: 42178,
    pc4: 1049,
    irish: 451,
    compas: 42193,
    vote: 56,
    cancer: 15,
    steel: 1504,
    kc1: 1067
  }
  task: {
    credit: 363626,
    transfusion: 363621,
    fitness: 363671,
    diabetes: 363629,
    biodegr: 363696,
    marketing: 363684
  }
```

| Parameter | Description |
|---|---|
| `type` | `'dataset'` — use OpenML dataset IDs; `'task'` — use OpenML task IDs (with predefined train/test splits) |
| `df_openml` | List of dataset names to run experiments on. Each name must exist as a key in the `dataset` or `task` dictionary |
| `dataset` | Mapping of dataset names → OpenML dataset IDs |
| `task` | Mapping of dataset names → OpenML task IDs |

**Available OpenML datasets:** `telco`, `pc4`, `tae`, `irish`, `compas`, `vote`, `cancer`, `steel`, `kc1`, `credit`, `transfusion`, `fitness`, `diabetes`, `biodegr`, `marketing`.

---

### Running with Custom Data

Set `DF_TYPE: 'custom'` in the `data` section:

```yaml
data:
  DF_TYPE: 'custom'
  DF_FORMAT: 'csv'
  LOCAL_DATASET_PATH: 'datasets/'
```

**Custom dataset requirements:**

1. Place your CSV files in the `datasets/` directory
2. Each CSV must have a target column named `label` (or `target` — it will be auto-renamed to `label`)
3. The target column must be **binary** (values: `0` and `1`)
4. For datasets not listed in LLMTabBench, define a task description in `source/prompts.py` in the `prompt_by_df` dictionary:

```python
prompt_by_df = {
    # ... existing entries ...
    'my_dataset': """For the given input features ... your task description ...
    Your output should be just a single number: 0 or 1. Do not predict any other tokens, only 0 or 1.""",
}
```

When `DF_TYPE` is `'custom'`, the code automatically discovers all files in `datasets/` and iterates over them.

---

### Experiment Settings — `experiment`

```yaml
experiment:
  baseline: False
  local_llm: True
  regime: 'local_gen'
  random_states_list: [864, 460, 142, 629, 761]
  serialization_list: ['feat_val', 'feat_val_masked', 'html', 'markdown', 'markdown_masked']
  shot_list: [4, 8, 16, 32, 64]
  RATIO: '50/50'
  N_SHOTS: 0
  SAMPLING_REGIME: 'default'
  RANDOM_STATE: 42
  SCHEMA: '{"class": "0 or 1"}'
  THINKING: False
  CONFIG_CODE: configuration_code
```

| Parameter | Description |
|---|---|
| `baseline` | `True` — run classical ML baselines; `False` — run LLM experiments |
| `local_llm` | `True` — use local vLLM inference; `False` — use API (OpenRouter) |
| `regime` | Inference mode: `'local_gen'` (local generation), `'local_nogen'` (forward pass only), `'api_gen'` (API generation) |
| `random_states_list` | List of random seeds; each experiment runs once per seed for stability |
| `serialization_list` | List of serialization formats to test (see below) |
| `shot_list` | List of few-shot sizes (e.g. `[4, 8, 16, 32, 64]`) |
| `RATIO` | Class ratio in few-shot examples, e.g. `'50/50'` for balanced |
| `SAMPLING_REGIME` | How to order few-shot examples: `'default'` (random), `'halves: zeros_first'`, `'halves: ones_first'`, `'cycle: zeros_first'`, `'cycle: ones_first'` |
| `SCHEMA` | JSON schema hint appended to the system prompt for local LLMs |
| `THINKING` | Enable/disable thinking mode in tokenizer chat template |
| `CONFIG_CODE` | Tag appended to output file names for experiment tracking |

---

### Available Serialization Formats

| Key | Format | Example |
|---|---|---|
| `feat_val` | Feature-value pairs | `Features are: age = 39, education = Bachelor. Answer is 0.` |
| `feat_val_masked` | Anonymized features | `Features are: x_1 = 39, x_2 = Bachelor. Answer is: y = 0.` |
| `feat_name` | Natural feature names | `Features are: the age is 39, the education is Bachelor. Answer is 0.` |
| `json` | JSON | `{"0": {"age": 39, "education": "Bachelor", "label": "0"}}` |
| `datamatrix` | Nested lists | `[['age', 'education', 'label'], [39, 'Bachelor', '0']]` |
| `markdown` | Markdown table | `\|age\|education\|label\|` |
| `markdown_masked` | Markdown (anonymized columns) | `\|feature_1\|feature_2\|label\|` |
| `html_new` | HTML table | `<table><thead>...</thead><tbody>...</tbody></table>` |
| `latex` | LaTeX table | `age & education & label \\` |
| `csv` | Comma-separated | `age, education, label` |
| `table` | Pipe-delimited | `\| 39 \| Bachelor \| 0 \|` |
| `dict` | Python dict | `{'age': [39], 'label': ['0']}` |
| `natural_language` | Natural language | Requires a `natural_language_data` column in the dataset |

---

### LLM Model Settings — `local_model`

```yaml
local_model:
  name: "Qwen/Qwen3-1.7B"
  temperature: 0
  gpu_memory_utilization: 0.12
  gpu_device: "0"
```

**Supported model families:**
- **Qwen**: e.g. `Qwen/Qwen3-1.7B`
- **Google Gemma**: e.g. `google/gemma-7b-it`
- **DeepSeek**: e.g. `deepseek-ai/deepseek-llm-7b-chat`

Each model family has its own prompt template and token handling logic.

---

### Baseline Model Settings — `baseline_model`

```yaml
baseline_model:
  name: 'logreg'
  all_names: ['logreg', 'gboost', 'rf']
```

When `experiment.baseline: True`, the benchmark runs classical ML models instead of LLMs.

| Model name | Algorithm |
|---|---|
| `logreg` | Logistic Regression (with Bayesian hyperparameter search) |
| `knn` | K-Nearest Neighbors |
| `rf` | Random Forest |
| `gboost` | Histogram-based Gradient Boosting (with early stopping) |
| `naive_argmax` | Dummy classifier (most frequent class) |

All baseline models (except `naive_argmax`) use **BayesSearchCV** from `scikit-optimize` for hyperparameter tuning.

---

## Usage Examples

### Example 1: Run LLM benchmark on OpenML datasets

```yaml
# config.yaml
data:
  DF_TYPE: 'openml'

experiment:
  baseline: False
  local_llm: True
  regime: 'local_gen'
  shot_list: [4, 8, 16]
  serialization_list: ['markdown_new', 'json_new']

local_model:
  name: "Qwen/Qwen3-1.7B"
  temperature: 0
  gpu_memory_utilization: 0.12
  gpu_device: "0"

openml:
  type: 'dataset'
  df_openml: ['compas', 'vote']
```

Then run:

```bash
make run
```

### Example 2: Run baselines on OpenML datasets

```yaml
# config.yaml
data:
  DF_TYPE: 'openml'

experiment:
  baseline: True
  shot_list: [4, 8, 16, 32, 64]

baseline_model:
  name: 'logreg'
  all_names: ['logreg', 'gboost', 'rf']

openml:
  type: 'dataset'
  df_openml: ['compas', 'vote', 'kc1']
```

### Example 3: Run on custom local datasets

1. Place your CSV files into `datasets/`:

```
datasets/
├── my_data.csv
└── another_data.csv
```

2. Add task descriptions in `source/prompts.py` (if you run datasets not listed in LLMTabBench):

```python
prompt_by_df = {
    ...
    'my_data': """For the given input features ... predict class 0 or 1. Do not predict any other tokens, only 0 or 1.""",
    'another_data': """For the given input features ... predict class 0 or 1. Do not predict any other tokens, only 0 or 1.""",
}
```

3. Update `config.yaml`:

```yaml
data:
  DF_TYPE: 'custom'
  DF_FORMAT: 'csv'
  LOCAL_DATASET_PATH: 'datasets/'
```

4. Run:

```bash
make run
```

---

## Output Structure

Results are saved in two directories:

### Metrics (`results/experiments_result/`)

```
results/experiments_result/
└── <dataset_name>/
    └── <N>_shots/
        └── <model_name>/
            └── rs_<regime>/
                ├── df_<N>fs_<model>_<regime>_<dataset>_<seed>_<config_code>.csv   # per-seed
                └── df_<N>fs_<model>_<regime>_<dataset>_<config_code>_agr.csv       # aggregated
```

The aggregated CSV contains `roc_auc_mean`, `roc_auc_std`, `f1_mean`, `f1_std`, `time_mean`, `time_std` across all random seeds.

### Probabilities (`results/llm_probs/`)

```
results/llm_probs/
└── <df_type>_datasets/
    └── <dataset_name>/
        └── <N>_shots/
            └── <model_name>/
                └── rs_<regime>/
                    └── df_<N>fs_<model>_<regime>_<dataset>_<serialization>_<seed>_<config_code>.pkl
```

Each `.pkl` file contains a dictionary with `pred_probs`, `true_labels`, `pred_labels`, `timestamp`, `params`, and metadata.

---

## Tips

- **GPU memory**: Adjust `local_model.gpu_memory_utilization` depending on your GPU. For small models like Qwen3-1.7B, `0.12` may be enough; for 7B models, increase to `0.45`+.
- **Reproducibility**: Results are aggregated across multiple random seeds defined in `random_states_list`. Using the same seeds ensures reproducibility.
- **Custom datasets**: Ensure your target column is binary (`0`/`1`). If the column is named `target`, it will be automatically renamed to `label`.
