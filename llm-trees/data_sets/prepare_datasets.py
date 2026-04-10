import os
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold


DATETIME_COLUMNS = {
    "bank_credit_scoring": ["BIRTHDATE"],
    "crimes_arrest": ["Date"],
}


def convert_datetime_columns(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    datetime_cols = DATETIME_COLUMNS.get(dataset_name, [])
    print(datetime_cols)

    for col in datetime_cols:
        if col not in df.columns:
            print(f"[WARN] {dataset_name}: datetime column '{col}' not found")
            continue
        dt = pd.to_datetime(df[col], errors="coerce")
        df[col] = dt.map(lambda x: x.timestamp() if pd.notna(x) else float("nan")).astype("float")

    return df


def get_real_df_local(name, base_path):
    path = os.path.join(base_path, f"{name}.csv")

    df = pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    if "Unnamed: 0.1" in df.columns:
        df = df.drop(columns=["Unnamed: 0.1"])

    if "target" in df.columns:
        df = df.rename(columns={"target": "label"})

    df = convert_datetime_columns(df, name)

    y = df["label"]
    X = df.drop(columns=["label"])

    rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=42)

    train_idx, test_idx = next(rskf.split(X, y))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    return X_train, y_train, X_test, y_test


def save_test_splits_to_repo(repo_base="ours_full"):
    dataset_names = [
        "bank_credit_scoring",
        "callcenter",
        "crimes_arrest",
        "extrovert",
        "machine",
        "postpartum",
        "reading",
        "stars",
    ]

    for dataset_name in dataset_names:
        dataset_dir = os.path.join(repo_base, dataset_name)

        try:
            _, _, X_test, y_test = get_real_df_local(
                name=dataset_name,
                base_path=dataset_dir,
            )

            x_out_path = os.path.join(dataset_dir, "X.csv")
            y_out_path = os.path.join(dataset_dir, "y.csv")
            
            X_test.to_csv(x_out_path, index=False)
            y_test.to_frame(name="label").to_csv(y_out_path, index=False)

            print(f"[OK] {dataset_name}:")
            print(f"     saved {x_out_path}")
            print(f"     saved {y_out_path}")

        except Exception as e:
            print(f"[ERROR] {dataset_name}: {e}")


if __name__ == "__main__":
    save_test_splits_to_repo()