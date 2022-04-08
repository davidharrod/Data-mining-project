import numpy as np
import pandas as pd
import sklearn.utils


def _read_csv(path):
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        print(f"There is no csv file in {path}.")
    return data


def _shuffle(data):
    return sklearn.utils.shuffle(data)


def _get_up_or_down(value):
    """If close price is greater or equal than open price, 
    then the the flag is set at 0. Otherwise, the flag is set at 1."""
    return 0 if value >= 0 else 1


def _get_prediction(data):

    df["Prediction"] = map(_get_up_or_down, df["Close"]-df["Open"])


def _divide_data(data, train_size):
    """Divide data to get training set and testing set."""
    data = _shuffle(data)
    return data.iloc[:train_size, :], data.iloc[train_size:, :]
