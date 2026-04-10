import os
import warnings

import pandas as pd
from cachetools import TTLCache
from cachetools import cached
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split

from .config import Config

warnings.simplefilter(action='ignore', category=FutureWarning)
imputing_neighbors = 10


def get_feature_types(dataset_name, X):
    # Default: no separate imputing for numeric and categorical features
    numerical_features = X.keys().to_list()
    categorical_features = []

    if dataset_name == "bank_credit_scoring":
        numerical_features = [
            "Debt",
            "Overdue days",
            "Initial limit",
            "INCOME",
            "TERM",
            "PDN",
            "UNDERAGECHILDRENCOUNT",
        ]
        categorical_features = [
            "SEX",
            "EDU",
            "Credit history rating",
            "LV_AREA",
            "LV_SETTLEMENTNAME",
            "INDUSTRYNAME",
            "FAMILYSTATUS",
            "BIRTHDATE"
        ]

    elif dataset_name == "callcenter":
        numerical_features = []
        categorical_features = X.keys().to_list()

    elif dataset_name == "crimes_arrest":
        numerical_features = [
            "Beat",
            "District",
            "Ward",
            "Community Area",
            "X Coordinate",
            "Y Coordinate",
            "Year",
            "Latitude",
            "Longitude",
        ]
        categorical_features = [
            "Primary Type",
            "Description",
            "Location Description",
            "Domestic",
            "FBI Code",
            "Date",
        ]

    elif dataset_name == "extrovert":
        numerical_features = []
        categorical_features = X.keys().to_list()

    elif dataset_name == "machine":
        numerical_features = [
            "Temperature",
            "Vibration",
            "Power_Usage",
            "Humidity",
        ]
        categorical_features = [
            "Machine_Type",
        ]

    elif dataset_name == "postpartum":
        numerical_features = [
            "Age",
        ]
        categorical_features = [
            "Number of household members",
            "Number of the latest pregnancy",
            "Residence",
            "Total children",
            "Education Level",
            "Marital status",
            "Occupation before latest pregnancy",
            "Monthly income before latest pregnancy",
            "Occupation After Your Latest Childbirth",
            "Current monthly income",
            "Husband's education level",
            "Husband's monthly income",
            "Addiction",
            "Disease before pregnancy",
            "History of pregnancy loss",
            "Family type",
            "Relationship with the in-laws",
            "Relationship with husband",
            "Relationship with the newborn",
            "Relationship between father and newborn",
            "Feeling about motherhood",
            "Recieved Support",
            "Need for Support",
            "Major changes or losses during pregnancy",
            "Abuse",
            "Trust and share feelings",
            "Pregnancy length",
            "Pregnancy plan",
            "Regular checkups",
            "Fear of pregnancy",
            "Diseases during pregnancy",
            "Age of newborn",
            "Age of immediate older children",
            "Mode of delivery",
            "Gender of newborn",
            "Birth compliancy",
            "Breastfeed",
            "Newborn illness",
            "Worry about newborn",
            "Relax/sleep when newborn is tended",
            "Relax/sleep when the newborn is asleep",
            "Angry after latest child birth",
            "Feeling for regular activities",
        ]

    elif dataset_name == "reading":
        numerical_features = [
            "Age",
            "Amount of time spent reading books per day",
        ]
        categorical_features = [
            "Number of e-books you read last two years",
            "Number of printed books you read last two years",
            "Gender",
            "Department",
            "Reason for reading books",
            "How to manage books for reading",
            "Favorite books for reading to spend leisure time",
            "What is the change that you fell after reading books",
            "Who influenced you to read books",
            "Which language books do you like to read",
            "Where do you read printed books",
            "Do you read Newspaper",
            "The format that you used for reading books",
            "Frequency of reading books",
        ]

    elif dataset_name == "stars":
        numerical_features = [
            "Vmag",
            "Plx",
            "e_Plx",
            "B-V",
            "Amag",
        ]
        categorical_features = [
            "SpType",
        ]

    return numerical_features, categorical_features


def load_and_split_data(config: Config):

    X, y = get_data(config.root, config.dataset)
    data_available = X is not None and y is not None
    if not data_available:
        raise FileNotFoundError(f"Data not found for dataset: {config.dataset}")

    # WE DON'T HAVE TRAIN PART
    # Init train test split
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, train_size=config.train_split, random_state=config.seed
    # )

    X_train, X_test, y_train, y_test = pd.DataFrame(), X, pd.Series(), y

    numerical_features, categorical_features = get_feature_types(config.dataset, X)
    X_train, X_test = impute_data(X_train, X_test, numerical_features, categorical_features)

    return X_train, X_test, y_train, y_test

@cached(cache=TTLCache(maxsize=1024, ttl=86400))
def get_data(root: str, dataset_name: str):

    try:
        if dataset_name == "acl":

            # the private ACL data is not included in the repository
            X = pd.read_csv(os.path.join(root, "data_sets/acl/X.csv"))
            y = pd.read_csv(os.path.join(root, "data_sets/acl/y.csv"))

            X["Group"] = X["Group"].replace("recon", 2)
            X["Group"] = X["Group"].replace("nc", 1)
            X["Group"] = X["Group"].replace("c", 0)

            X["Sex"] = X["Sex"].replace("female", 0)
            X["Sex"] = X["Sex"].replace("male", 1)

            X["Dominant_Leg"] = X["Dominant_Leg"].replace("left", 0)
            X["Dominant_Leg"] = X["Dominant_Leg"].replace("right", 1)

            # rename keys to match the other datasets
            X = X.rename(columns={
                " ccMF.D.T2.Me": "ccMF.D.T2.Me",
                " ccMF.S.T2.Me": "ccMF.S.T2.Me",
                "  Age": "Age",
            })

            # Mapping [-1  1] to [0 1] for the XGBoost model
            y = y > 0

        else:
            path = os.path.join(root, f"data_sets/{dataset_name}")
            X = pd.read_csv(os.path.join(path, "X.csv"))
            y = pd.read_csv(os.path.join(path, "y.csv"))

        return X, y

    except FileNotFoundError:
        return None, None


def impute_data(X_train, X_test, numerical_features=None, categorical_features=None):

    if numerical_features is not None:
        for col in numerical_features:
            if X_test is not None:
                X_test[col] = X_test[col].astype(float)

        # WE DON'T HAVE TRAIN PART
        # if numerical_features:
        #     mean_imputer = KNNImputer(n_neighbors=imputing_neighbors, keep_empty_features=True)
        #     X_train[numerical_features] = mean_imputer.fit_transform(X_train[numerical_features])
        #     if X_test is not None:
        #         X_test[numerical_features] = mean_imputer.transform(X_test[numerical_features])

        # if categorical_features:
        #     mode_imputer = SimpleImputer(strategy='most_frequent', keep_empty_features=True)
        #     X_train[categorical_features] = mode_imputer.fit_transform(X_train[categorical_features])
        #     if X_test is not None:
        #         X_test[categorical_features] = mode_imputer.transform(X_test[categorical_features])

        # cast categorical features to int
        # for col in categorical_features:
        #     # X_train[col] = X_train[col].astype(int)
        #     if X_test is not None:
        #         X_test[col] = X_test[col].astype(int)

    else:
        # imputer = KNNImputer(n_neighbors=n_neighbors)
        imputer = SimpleImputer(strategy='most_frequent')
        X_train = imputer.fit_transform(X_train)
        if X_test is not None:
            X_test = imputer.transform(X_test)

    return X_train, X_test
