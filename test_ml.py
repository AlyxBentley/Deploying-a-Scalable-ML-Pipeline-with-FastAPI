import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics


# Constants used across tests
DATA_PATH = os.path.join("data", "census.csv")

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
LABEL = "salary"


def _small_processed_split(sample_n=2000, test_size=0.2, seed=42):
    """Helper to create a small deterministic train/test split and process it."""
    df = pd.read_csv(DATA_PATH).sample(n=sample_n, random_state=seed).reset_index(drop=True)
    train, test = train_test_split(df, test_size=test_size, random_state=seed)

    X_train, y_train, encoder, lb = process_data(
        X=train,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=True,
    )

    X_test, y_test, _, _ = process_data(
        X=test,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    return X_train, y_train, X_test, y_test


# TODO: implement the first test. Change the function name and input as needed
def test_one():
    """
    Test that process_data returns arrays with matching row counts
    and consistent feature dimensions.
    """

    X_train, y_train, X_test, y_test = _small_processed_split()

    assert hasattr(X_train, "shape")
    assert hasattr(X_test, "shape")

    # same number of encoded features in train and test
    assert X_train.shape[1] == X_test.shape[1]

    # y lengths match X rows
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]


# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    Test that train_model returns a LogisticRegression model.
    """
    X_train, y_train, _, _ = _small_processed_split()
    model = train_model(X_train, y_train)

    assert isinstance(model, LogisticRegression)


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    Test that inference + compute_model_metrics return valid outputs.
    """
    X_train, y_train, X_test, y_test = _small_processed_split()
    model = train_model(X_train, y_train)

    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_test.shape[0]

    precision, recall, f1 = compute_model_metrics(y_test, preds)

    for m in (precision, recall, f1):
        assert isinstance(m, float)
        assert 0.0 <= m <= 1.0
