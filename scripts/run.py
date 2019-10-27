import pandas as pd
import numpy as np

from implementations import *
from helpers import compute_accuracy, predict_labels, load_csv_data, create_csv_submission, split_data
import experiments

DATA_TRAIN_PATH = "../data/train.csv"
DATA_TEST_PATH = '../data/test.csv'
OUTPUT_PATH = '../submission.csv'


def preprocess_data(tX, y):
    tX_new, y_new, _ = experiments.preprocessing_option_9(tX, y)
    return tX_new, y_new


def main():
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

    np.random.seed(2019)

    tX_prep, y_prep = preprocess_data(tX, y)
    tX_test_prep, _ = preprocess_data(tX_test, None)

    lambda_ = 1
    w, _ = ridge_regression(y_prep, tX_prep, lambda_)

    y_pred = predict_labels(w, tX_test_prep)
    print(np.unique(y_pred, return_counts=True))

    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)


if __name__ == "__main__":
    main()
