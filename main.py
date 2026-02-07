import os
import yaml
from source.models import run_fewshot_iteration, load_model_and_tokenizer


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
        

dataset_files = [f.lower().split('.')[0] for f in os.listdir(config['data']['LOCAL_DATASET_PATH'])]
  

def main(config):

    baseline = config['experiment']['baseline']
    shot_list = config['experiment']['shot_list'] 
    type_df = config['data']['DF_TYPE']

    if not baseline:
        model, tokenizer = load_model_and_tokenizer(config)

    if type_df == 'openml':
        ds_iterate = config['openml']['df_openml']
    else:
        ds_iterate = dataset_files

    for ds in ds_iterate:
        config['data']['DATASET_NAME'] = ds
        for s in shot_list:
            config['experiment']['N_SHOTS'] = s
            if baseline:
                for model_name in config['baseline_model']['all_names']:
                    config['baseline_model']['name'] = model_name
                    df = run_fewshot_iteration(config)
            else:
                df = run_fewshot_iteration(config, model, tokenizer)


if __name__ == '__main__':
    main(config)