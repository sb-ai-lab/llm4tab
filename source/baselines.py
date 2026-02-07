import numpy as np
import os

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier


from datetime import datetime
from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV
from sklearn.metrics import roc_auc_score, f1_score
import time
import pandas as pd
import warnings
warnings.filterwarnings("ignore")



# Logistic Regression
def LR(X, y, n_iter=10, cv=20):
    """ Logistic Regression

        # Arguments:
            X:          Training/Validation data matrix with inputs of size (N x Nx), Nx - number of inputs.
            y:          Training/Validation data matrix with outputs of size (N x Ny), Ny - number of outputs.
            n_iter:    The number of GP function evaluations.
    """
    if X.shape[0] > 256:
        opt = BayesSearchCV(
            LogisticRegression(),
            {
                'solver': Categorical(['newton-cg', 'lbfgs', 'sag']),
                'C': Real(1e-3, 10, prior='log-uniform')
            },
            n_iter=n_iter,
            random_state=0,
            cv=cv,
            iid = False,
            n_jobs = -1,
        )
    else:
        opt = LogisticRegression()
    model = opt.fit(X, y)

    return model


# K-Near Neighbors
def KNN(X, y, n_iter=10, cv=20, max_neighbours=20):
    """ K-Neighbors for Classification

        # Arguments:
            X:          Training/Validation data matrix with inputs of size (N x Nx), Nx - number of inputs.
            y:          Training/Validation data matrix with outputs of size (N x Ny), Ny - number of outputs.
            n_iter:    The number of GP function evaluations.
    """
    opt = BayesSearchCV(
        KNeighborsClassifier(),
        {
            'n_neighbors': Integer(1, max_neighbours),
            'weights': Categorical(['uniform', 'distance']),
        },
        n_iter=n_iter,
        random_state=0,
        cv=cv,
        iid = False,
        n_jobs = -1,
    )
    model = opt.fit(X, y)

    return model


# Random Forest
def RF(X, y, n_iter=10, cv=20):
    ''' Random Forest for Classification

        # Arguments:
            X:          Training/Validation data matrix with inputs of size (N x Nx), Nx - number of inputs.
            y:          Training/Validation data matrix with outputs of size (N x Ny), Ny - number of outputs.
            n_iter:    The number of GP function evaluations.
    '''
    opt = BayesSearchCV(
        RandomForestClassifier(),
        {
            'n_estimators': Integer(50, 1000),
            'max_depth': Integer(3, 15),
            'min_samples_leaf': Integer(3, 15),
            'min_samples_split': Integer(3, 15)
        },
        n_iter=n_iter,
        random_state=0,
        cv=cv,
        iid = False,
        n_jobs = -1,
    )
    model = opt.fit(X, y)

    return model


    
# Extreme Gradient Boosting
def XGB(X, y, n_iter=10, cv=20):
    ''' Extreme Gradient Boosting for Classification

        # Arguments:
            X:          Training/Validation data matrix with inputs of size (N x Nx), Nx - number of inputs.
            y:          Training/Validation data matrix with outputs of size (N x Ny), Ny - number of outputs.
            n_iter:     The number of GP function evaluations.
    '''

    opt = BayesSearchCV(
        HistGradientBoostingClassifier(),
        {
            "max_depth": Integer(3, 15),
            "learning_rate": Real(1e-3, 1, prior="log-uniform"),
            "min_samples_leaf": Integer(1, 20),
            "l2_regularization": Real(1e-6, 10.0, prior="log-uniform"),
            "max_leaf_nodes": Integer(7, 63),
        },
        n_iter=n_iter,
        random_state=0,
        cv=cv,
        iid = False,
        n_jobs = -1,
    )
    model = opt.fit(X, y)

    return model



def Naive(X, y):
    ''' DummyClassifier for Classification

        # Arguments:
            X:          Training/Validation data matrix with inputs of size (N x Nx), Nx - number of inputs.
            y:          Training/Validation data matrix with outputs of size (N x Ny), Ny - number of outputs.
    '''
    opt = DummyClassifier(strategy="most_frequent")
    model = opt.fit(X, y)

    return model





def get_model_config(model_name):
    configs = {
        "logreg": {
            "estimator": LogisticRegression,
            "params": {
                "solver": Categorical(["newton-cg", "lbfgs", "sag"]),
                "C": Real(1e-3, 10, prior="log-uniform"),
            },
            "use_bayes": lambda X: X.shape[0] >= 256,
        },
        "knn": {
            "estimator": KNeighborsClassifier,
            "params": {
                "n_neighbors": Integer(1, 20),
                "weights": Categorical(["uniform", "distance"]),
            },
            "use_bayes": lambda X: True,
            "max_neighbors": lambda n_shots: 2 if n_shots < 10 else min(20, int(n_shots / 2)),
        },
        "rf": {
            "estimator": RandomForestClassifier,
            "params": {
                "n_estimators": Integer(50, 1000),
                "max_depth": Integer(3, 15),
                "min_samples_leaf": Integer(3, 15),
                "min_samples_split": Integer(3, 15),
            },
            "use_bayes": lambda X: True,
        },
        "gboost": {
            "fixed_params": {
                "max_iter": 10000,
                "early_stopping": True,
                "n_iter_no_change": 100,
                "validation_fraction": None,
                "random_state": 42,
            },
            "estimator": HistGradientBoostingClassifier,
            "params": {
                "max_depth": Integer(3, 15),
                "learning_rate": Real(1e-3, 1, prior="log-uniform"),
                "min_samples_leaf": Integer(1, 20),
                "l2_regularization": Real(1e-6, 10.0, prior="log-uniform"),
                "max_leaf_nodes": Integer(7, 63),
            },
            "use_bayes": lambda X: True,
        },
        "naive_argmax": {
            "fixed_params": {"strategy": "most_frequent"},
            "estimator": DummyClassifier,
            "params": {},
            "use_bayes": lambda X: False,
        },
    }
    return configs[model_name]