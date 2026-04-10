import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, MinMaxScaler

from .config import Config
from .utils import predict_tree_from_file, regenerate_tree, get_tree_path, generate_tree


def get_preprocessor(config, numerical_features, categorical_features):
    """
    Returns the appropriate preprocessor based on the classifier type.

    Parameters:
        config (Config): Configuration object containing classifier.
        numerical_features (list): List of numerical feature names.
        categorical_features (list): List of categorical feature names.

    Returns:
        ColumnTransformer or FunctionTransformer: A transformer for preprocessing.
    """
    return ColumnTransformer([
        ("nominal", Pipeline([
            ("nominal_imputer", SimpleImputer(strategy="most_frequent")),
            ("nominal_encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features),
        ("numerical", Pipeline([
            ("numerical_imputer", KNNImputer(n_neighbors=10, keep_empty_features=True)),
            ("numerical_scaler", get_numerical_scaler(config))
        ]), numerical_features),
    ], sparse_threshold=0)


def get_numerical_scaler(config):
    """
    Returns the appropriate preprocessor based on the classifier type.

    Parameters:
        config (Config): Configuration object containing classifier.

    Returns:
        FunctionTransformer: A transformer for scaling.
    """
    return MinMaxScaler()


def fit_classifier(config, transformation, X, y):
    """
    Fit a classifier using a pipeline with preprocessing and grid search.
    """

    check_scaling(config, transformation, X)

    if config.classifier == "mlp":
        classifier = MLPClassifier(
            solver="lbfgs",
            max_iter=10000,
            random_state=config.seed,
        )
        param_grid = {
            "classifier__hidden_layer_sizes": [10, 25, 50, 75, 100],
            "classifier__alpha": [0.0001, 0.001, 0.01, 0.1, 1],
        }

    elif config.classifier == "hgbdt":
        classifier = HistGradientBoostingClassifier(
            random_state=config.seed,
        )
        param_grid = {
            "classifier__min_samples_leaf": [5, 10, 15, 20, 25]
        }

    elif config.classifier == "lr":
        classifier = LogisticRegression(
            max_iter=10000,
            random_state=config.seed,
        )
        param_grid = {
            "classifier__C": [1, 10, 100, 1000, 10000]
        }

    pipeline = Pipeline([
        ("transformer", transformation),
        ("classifier", classifier)
    ])

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
    )

    search.fit(X, y)
    return search.best_estimator_

def check_scaling(config, transformation, X):
    X_transformed = transformation.fit_transform(X)
    x_min = np.min(X_transformed)
    x_max = np.max(X_transformed)
    if x_min < -0.1 or x_max > 1.1:
        raise ValueError(f"Invalid range: [{x_min}, {x_max}] (method: {config.method})")


def tree_to_embedding(X: pd.DataFrame, config: Config):
    """
    Generate embeddings from decision trees.

    # Source: Borisov, V.; Broelemann, K.; Kasneci, E.; and Kasneci, G. 2023.
    # DeepTLF: robust deep neural networks for heterogeneous tabular data.
    # International Journal of Data Science and Analytics, 16(1): 85–100.

    """

    embedding = np.zeros((len(X), 0))
    for tree_idx in range(config.num_trees):
        config.iter = tree_idx

        generate_tree(config)

        for try_idx in range(config.num_retry_llm + 1):
            if try_idx == config.num_retry_llm:
                raise ValueError(f"Reached maximum number of retries for tree: {get_tree_path(config)}")
            try:
                # Predict the tree
                _, nodes = predict_tree_from_file(config, X)

                # Collect the embeddings
                embedding = np.hstack((embedding, nodes))

                # If the tree was successfully predicted, move on to the next tree
                break
            except Exception as e:
                regenerate_tree(config, e)

    return np.array(embedding)
