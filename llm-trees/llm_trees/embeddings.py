import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBRegressor, XGBClassifier

from .config import Config
from .embedding_utils import get_preprocessor, fit_classifier, tree_to_embedding, get_numerical_scaler
from .io import load_and_split_data, get_feature_types
from .metrics import score
from .utils import generate_tree


def eval_embedding(config):

    try:

        # Generate trees if necessary
        if config.method in ['gpt-4o-mini', "gpt-4o", "gpt-o1", "gemini", "claude"]:
            for tree_idx in range(config.num_trees):
                config.iter = tree_idx
                generate_tree(config)

        X_train, X_test, y_train, y_test = load_and_split_data(config)

        if config.method == "no":
            pipeline = wo_embedding(X_train, y_train.values.ravel(), config)
        elif config.method == "claude":
            pipeline = llm_embedding(X_train, y_train.values.ravel(), config)
        elif config.method == "gemini":
            pipeline = llm_embedding(X_train, y_train.values.ravel(), config)
        elif config.method == "gpt-4o-mini":
            pipeline = llm_embedding(X_train, y_train.values.ravel(), config)
        elif config.method == "gpt-4o":
            pipeline = llm_embedding(X_train, y_train.values.ravel(), config)
        elif config.method == "gpt-o1":
            pipeline = llm_embedding(X_train, y_train.values.ravel(), config)
        elif config.method == "rt-us":
            pipeline = rt_embedding(X_train, y_train.values.ravel(), config)
        elif config.method == "et-sv":
            pipeline = et_embedding(X_train, y_train.values.ravel(), config, False)
        elif config.method == "et-ss":
            pipeline = et_embedding(X_train, y_train.values.ravel(), config, True)
        elif config.method == "rf-sv":
            pipeline = rf_embedding(X_train, y_train.values.ravel(), config, False)
        elif config.method == "rf-ss":
            pipeline = rf_embedding(X_train, y_train.values.ravel(), config, True)
        elif config.method == "xg-sv":
            pipeline = xgb_embedding(X_train, y_train.values.ravel(), config, False)
        elif config.method == "xg-ss":
            pipeline = xgb_embedding(X_train, y_train.values.ravel(), config, True)
        else:
            raise ValueError(f"Unknown method: {config.method}")

        # Compute the scores
        acc = score(pipeline, X_test, y_test.values.ravel(), "acc")
        f1 = score(pipeline, X_test, y_test.values.ravel(), "f1")

    except Exception as e:
        print(f"Scoring failed: {e}")
        acc, f1 = -1, -1

    return acc, f1

def wo_embedding(X, y, config: Config):
    """
    Fit a linear probe classifier directly to raw features without additional embedding.
    """
    numerical_features, categorical_features = get_feature_types(config.dataset, X)
    preprocessor = get_preprocessor(config, numerical_features, categorical_features)
    return fit_classifier(config, preprocessor, X, y)


def llm_embedding(X, y, config):

    if config.append_raw_features:

        # Step 1: Generate embeddings
        generate_embeddings = FunctionTransformer(lambda X: tree_to_embedding(X, config))

        # Step 2: Preprocess raw data
        numerical_features, categorical_features = get_feature_types(config.dataset, X)
        preprocessor = get_preprocessor(config, numerical_features, categorical_features)

        # Combine both transformations
        embedding_union = FeatureUnion([
            ('embeddings', generate_embeddings),
            ('preprocessed', preprocessor)
        ])

        embedding_transformation = embedding_union
        # # Apply StandardScaler after embedding_union
        # embedding_transformation = Pipeline([('embedding_union', embedding_union), ('scaler', StandardScaler())])

    else:
        embedding_transformation = FunctionTransformer(lambda X: tree_to_embedding(X, config))

    return fit_classifier(config, embedding_transformation, X, y)


def rt_embedding(X: pd.DataFrame, y: np.array, config: Config):
    # unsupervised random trees embedding (Moosmann et al., 2006
    # (https://proceedings.neurips.cc/paper/2006/hash/d3157f2f0212a80a5d042c127522a2d5-Abstract.html))
    max_depth = 5 if config.max_tree_depth == 0 else config.max_tree_depth

    random_trees_embedding = ensemble.RandomTreesEmbedding(
        n_estimators=config.num_trees,
        max_depth=max_depth,
        random_state=config.seed,
    ).fit(X)

    embedding_transformation = FunctionTransformer(
        lambda X: random_trees_embedding.transform(X).toarray()
    )

    return fit_classifier(config, embedding_transformation, X, y)


def et_embedding(X, y, config, self_supervised=False):
    # self-supervised extra-trees embedding (Borisov et al., 2023
    # (https://link.springer.com/article/10.1007/s41060-022-00350-z))

    max_depth = None if config.max_tree_depth == 0 else config.max_tree_depth

    if self_supervised:
        model = ensemble.ExtraTreesRegressor(
            n_estimators=config.num_trees,
            max_depth=max_depth,
            random_state=config.seed,
        ).fit(X, X)
    else:
        model = ensemble.ExtraTreesClassifier(
            n_estimators=config.num_trees,
            max_depth=max_depth,
            random_state=config.seed,
        ).fit(X, y)

    embedding = FunctionTransformer(lambda X: model.apply(X))
    scaler = get_numerical_scaler(config)
    transformation = Pipeline([('embedding', embedding), ('scaler', scaler)])

    return fit_classifier(config, transformation, X, y)


def rf_embedding(X, y, config, self_supervised=False):
    # unsupervised random trees embedding (Moosmann et al., 2006
    # (https://proceedings.neurips.cc/paper/2006/hash/d3157f2f0212a80a5d042c127522a2d5-Abstract.html))
    max_depth = None if config.max_tree_depth == 0 else config.max_tree_depth
    if self_supervised:
        model = ensemble.RandomForestRegressor(
            n_estimators=config.num_trees,
            max_depth=max_depth,
            random_state=config.seed,
        ).fit(X, X)
    else:
        model = ensemble.RandomForestClassifier(
            n_estimators=config.num_trees,
            max_depth=max_depth,
            random_state=config.seed,
        ).fit(X, y)

    embedding_transformation = FunctionTransformer(lambda X: model.apply(X))
    scaler = get_numerical_scaler(config)
    transformation = Pipeline([('embedding', embedding_transformation), ('scaler', scaler)])

    return fit_classifier(config, transformation, X, y)


def xgb_embedding(X, y, config, self_supervised=False):
    # self-supervised XGBoost embedding (Borisov et al., 2023
    # (https://link.springer.com/article/10.1007/s41060-022-00350-z))
    max_depth = None if config.max_tree_depth == 0 else config.max_tree_depth
    if self_supervised:
        model = XGBRegressor(
            n_estimators=config.num_trees,
            max_depth=max_depth,
            random_state=config.seed,
        ).fit(X, X)
    else:
        model = XGBClassifier(
            n_estimators=config.num_trees,
            max_depth=max_depth,
            random_state=config.seed,
        ).fit(X, y)

    def transform_func(X):
        X_transformed = model.apply(X)
        if X_transformed.ndim == 1:
            X_transformed = X_transformed.reshape(-1, 1)
        return X_transformed

    embedding_transformation = FunctionTransformer(lambda X: transform_func(X))

    scaler = get_numerical_scaler(config)
    transformation = Pipeline([('embedding', embedding_transformation), ('scaler', scaler)])

    return fit_classifier(config, transformation, X, y)
