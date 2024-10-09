import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os


def load_and_preprocess_data(data_path):
    """Loads, preprocesses, and splits the dataset."""

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No such file or directory: '{data_path}'")

    index_column = [
        "White King file",
        "White King rank",
        "White Rook file",
        "White Rook rank",
        "Black King file",
        "Black King rank",
        "Target",
    ]

    Win_move_vs_draw = pd.read_csv(data_path, sep=",", header=None, names=index_column)

    class_labels = Win_move_vs_draw["Target"].unique()
    categorical_columns = Win_move_vs_draw.select_dtypes(include=["object"]).columns

    # Convert categorical columns to numeric categories
    Win_move_vs_draw[categorical_columns] = (
        Win_move_vs_draw[categorical_columns]
        .astype("category")
        .apply(lambda x: x.cat.codes)
    )

    df = Win_move_vs_draw
    X = df.drop("Target", axis=1)
    y = df["Target"]
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    # Use the shuffled indices for splitting the data
    X = X.iloc[indices]
    X = pd.get_dummies(X, columns=X.select_dtypes(include=["object"]).columns)
    y = y[indices]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    return X_train, X_test, y_train, y_test, class_labels


if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test, class_labels = load_and_preprocess_data(
            "data/raw/krkopt.DATA"
        )
    except FileNotFoundError as e:
        print(e)
        # You can save the processed data here if needed (e.g., to data/processed)
