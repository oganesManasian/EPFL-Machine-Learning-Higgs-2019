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

    tX_stacked = np.vstack((tX, tX_test))
    # Preprocess data together to have the same shifts while creating log or root features
    tX_stacked_prep, _ = preprocess_data(tX_stacked, None)
    tX_prep, tX_test_prep = np.split(tX_stacked_prep, [len(tX)])

    lambda_ = 0
    w, _ = ridge_regression(y, tX_prep, lambda_)

    y_pred = predict_labels(w, tX_test_prep)
    print(np.unique(y_pred, return_counts=True))

    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)


if __name__ == "__main__":
    main()
    main()
