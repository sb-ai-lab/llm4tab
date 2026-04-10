import os.path

from .config import Config
from .prompt_examples import get_example


def get_role_prompt():
    return f"""
You are a domain expert with years of experience in building the best-performing decision trees. 
You have an astounding ability to identify the best features for the task at hand, and you know how to combine them to get the best predictions. 
Impressively, your profound world knowledge allows you to do that without looking at any real-world data.

"""

def get_intro(config: Config):

    if config.max_tree_depth:
        tree_string =  f" with a maximum depth of {config.max_tree_depth}"
        tree_string += f" (it can have between {config.max_tree_depth} and {2 ** (config.max_tree_depth) - 1} inner nodes)"
    else:
        tree_string = ""

    return f"""
I want you to induce a decision tree classifier based on features and a prediction target. 
I first give {config.num_examples} examples below. Given Features and a new prediction target, I then want you to build a decision tree{tree_string} using the most important features. 
Format the decision tree as a Python function that returns a single prediction as well as a list representing the truth values of the inner nodes. 
The entries of this list should be 1 if the condition of the corresponding inner node is satisfied, and 0 otherwise. 
Use only the feature names that I provide, generate the decision tree without training it on actual data, and return the Python function.
"""



def get_free_intro():
    return f"""
I want you to induce a classifier based on features and a prediction target. 
Format the classifier as a Python function called 'predict' that takes a dictionary with the features as input and returns a single integer. 
Use only the feature names that I provide, generate the classifier without training it on actual data, and return only the Python function without further explanation.

"""

def get_features(config: Config):
    feature_txt_file = os.path.join(config.root, "data_sets", config.dataset, "feature_description.txt")
    with open(feature_txt_file, "r") as f:
        features = f.read()
    return features


def get_description(config):
    feature_txt_file = os.path.join(config.root, "data_sets", config.dataset, "description.txt")
    with open(feature_txt_file, "r") as f:
        description = f.read()

    if description != "":
        return f"Describtion of the dataset: \n {description}"
    else:
        return ""


def get_end_part():
    return """
Decision tree:
def predict(X: dict):
"""

def get_free_end_part():
    return """
def predict(X: dict) -> int:
"""


def get_full_prompt(config: Config) -> str:
    prompt = ""
    if config.use_role_prompt:
        prompt += get_role_prompt()
    prompt += get_intro(config)
    for i in range(config.num_examples):
        prompt += get_example(i, config)
    prompt += get_features(config) + "\n"
    if config.include_description:
        prompt += get_description(config) + "\n"
    prompt += get_end_part()
    return prompt


def get_free_prompt(config: Config):
    prompt = ""
    if config.use_role_prompt:
        prompt += get_role_prompt()
    prompt += get_free_intro()
    prompt += get_features(config) + "\n"
    if config.include_description:
        prompt += get_description(config) + "\n"
    prompt += get_free_end_part()
    return prompt


def get_first_prompt(config: Config):
    prompt = ""

    if config.use_role_prompt:
        prompt += get_role_prompt()

    prompt_path = os.path.join(config.root, "data_sets", config.dataset, "prompt.txt") 
    with open(prompt_path, "r") as f:
        prompt += f.read()

    if config.num_examples > 1:
        prompt = prompt.replace("an example below", f"{config.num_examples} examples below")

    if config.max_tree_depth != 2:
        if config.max_tree_depth > 0:
            prompt = prompt.replace(" with a maximum depth of 2", f" with a maximum depth of {config.max_tree_depth}")
        else:
            prompt = prompt.replace(" with a maximum depth of 2", f"")

    return prompt


def get_second_prompt(config: Config):
    second_prompt = """
Now, format this decision tree as a Python function that returns a single prediction as well as a list representing the truth values of the inner nodes. 
The entries of this list should be 1 if the condition of the corresponding inner node is satisfied, and 0 otherwise. 

If your decision tree has placeholders for thresholds, try to fill them with meaningful values. 

"""
    feature_string = get_feature_string(config)
    second_prompt += feature_string

    if config.num_examples == 1:
        second_prompt += "Here is an example of what the function should look like:"
    else:
        second_prompt += f"Here are {config.num_examples} examples of what the function should look like:"

    for i in range(config.num_examples):
        second_prompt += get_example(i, config)

    return second_prompt


def get_feature_string(config):
    keys, target, feature_description = get_feature_keys(config)

    feature_string = f"""
        
Here are the exact keys of the features. Pay close attention to their spelling (uppercase, lowercase, spaces, etc.):
{keys}

Here is the target variable. Pay close attention to the target value encoding:
{target}

Here is a description of the features with optional feature encoding or ranges:
{feature_description}
"""
    return feature_string


def get_feature_keys(config):
    features_str = get_features(config)
    # iterate line by linea over the features string:
    keys = []
    target_line_idx = -1
    target = ""
    feature_description = ""
    for line_idx, line in enumerate(features_str.split("\n")):
        if line_idx == 0:
            continue

        if line_idx == target_line_idx + 1:
            target = line
            break

        # if no : in the line or the line contains "target variable", skip
        if ":" not in line:
            continue

        if "target variable" in line:
            target_line_idx = line_idx
            continue

        # get the key of the feature (i.e. everything befor the :)
        key = line.split(":")[0]
        keys.append(key)
        feature_description += line + "\n"


    return keys, target, feature_description
