from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from skopt.space import Real, Integer, Categorical


def logreg_model():
    return {
        "estimator": LogisticRegression,
        "params": {
            "solver": Categorical(["newton-cg", "lbfgs", "sag"]),
            "C": Real(1e-3, 10, prior="log-uniform"),
        },
        "use_bayes": lambda X: X.shape[0] >= 256,
    }


def knn_model(n_shots):
    max_neighbors = 2 if n_shots < 10 else min(20, int(n_shots / 2))

    return {
        "estimator": KNeighborsClassifier,
        "params": {
            "n_neighbors": Integer(1, max_neighbors),
            "weights": Categorical(["uniform", "distance"]),
        },
        "use_bayes": lambda X: True,
    }


def rf_model():
    return {
        "estimator": RandomForestClassifier,
        "params": {
            "n_estimators": Integer(50, 1000),
            "max_depth": Integer(3, 15),
            "min_samples_leaf": Integer(3, 15),
            "min_samples_split": Integer(3, 15),
        },
        "use_bayes": lambda X: True,
    }


def gboost_model(n_shots):
    fixed_params = {
        "early_stopping": True,
        "n_iter_no_change": 100,
        "random_state": 42,
    }

    if n_shots <= 4:
        fixed_params["validation_fraction"] = None
        search_space = {
            "max_iter": Integer(1, 1000),
            "max_depth": Integer(3, 15),
            "learning_rate": Real(1e-3, 1, prior="log-uniform"),
            "min_samples_leaf": Integer(1, 20),
            "l2_regularization": Real(1e-6, 10.0, prior="log-uniform"),
            "max_leaf_nodes": Integer(7, 63),
        }
    else:
        fixed_params["validation_fraction"] = (
            0.3 if n_shots <= 32 else 0.2 if n_shots <= 128 else 0.1
        )
        search_space = {
            "max_depth": Integer(3, 15),
            "learning_rate": Real(1e-3, 1, prior="log-uniform"),
            "min_samples_leaf": Integer(1, 20),
            "l2_regularization": Real(1e-6, 10.0, prior="log-uniform"),
            "max_leaf_nodes": Integer(7, 63),
        }

    return {
        "estimator": HistGradientBoostingClassifier,
        "fixed_params": fixed_params,
        "params": search_space,
        "use_bayes": lambda X: True,
    }


def naive_model():
    return {
        "estimator": DummyClassifier,
        "fixed_params": {"strategy": "most_frequent"},
        "params": {},
        "use_bayes": lambda X: False,
    }


def get_baseline_model(model_name, n_shots):
    models = {
        "logreg": lambda: logreg_model(),
        "knn": lambda: knn_model(n_shots),
        "rf": lambda: rf_model(),
        "gboost": lambda: gboost_model(n_shots),
        "naive_argmax": lambda: naive_model(),
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    return models[model_name]()