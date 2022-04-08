import pandas as pd
import sklearn.utils
from sklearn import model_selection


def _read_csv(path):
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        print(f"There is no csv file in {path}.")
        return None
    return data


def _shuffle(data):
    # The index numbers will be out of order.
    return sklearn.utils.shuffle(data)


def _get_up_or_down(value):
    """If close price is greater or equal than open price, 
    then the the flag is set at 0. Otherwise, the flag is set at 1."""
    return 0 if value >= 0 else 1


def _get_prediction(data):
    data["Prediction"] = list(map(_get_up_or_down, data["Close"]-data["Open"]))
    return data


def _drop_unrelative_keys(data):
    return data.drop(["Date", "Adj Close"], axis=1)


def _split_data(data, test_size=0.25):
    """Divide data to get training set and testing set.
    Return X_train,X_test,y_train,y_test"""
    X = _drop_unrelative_keys(data)
    data = _get_prediction(data)
    y = data.Prediction
    return model_selection.train_test_split(X, y, test_size=test_size, random_state=1234)


def load_data(path, test_size=0.25):
    data = _read_csv(path)
    return _split_data(data, test_size)


# Test codes.
# path = "./data/data.csv"
# load_data(path)
