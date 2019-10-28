import pandas as pd
import numpy as np

from implementations import ridge_regression
from helpers import compute_accuracy, predict_labels, load_csv_data, create_csv_submission, split_data, lambda_cv
from preprocessing_data import preprocess_data, divide_data

DATA_TRAIN_PATH = "../data/train.csv"
DATA_TEST_PATH = '../data/test.csv'
OUTPUT_PATH = '../submission.csv'


def main():
    y_train, tX_train, ids = load_csv_data(DATA_TRAIN_PATH)
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

    np.random.seed(2019)

    # Preprocess data together to have the same shifts while creating log or root features
    tX_stacked = np.vstack((tX_train, tX_test))
    prep_param = {"bias": True, "fill": True, "standardize": False, "degree": 8, "log": True, "root": True}
    tX_stacked_prep, *_ = preprocess_data(tX_stacked, None, prep_param)
    tX_train_prep, tX_test_prep = np.split(tX_stacked_prep, [len(tX_train)])

    # Split data according to PRI_jet_num value
    tX_tr_splitted, indices_tr = divide_data(tX_train_prep)
    tX_te_splitted, indices_te = divide_data(tX_test_prep)
    n_models = len(indices_tr)

    y_tr_splitted = []
    for i in range(n_models):
        y_tr_splitted.append(y_train[indices_tr[i]])

    # Train
    weights = []
    for i in range(n_models):
        lambda_ = lambda_cv(tX_tr_splitted[i], y_tr_splitted[i])
        print(f"Class {i}, lambda: {lambda_}")
        weights.append(ridge_regression(y_tr_splitted[i], tX_tr_splitted[i], lambda_)[0])

    # Predict
    y_pr_tr = np.zeros(tX_train.shape[0])
    y_pr_te = np.zeros(tX_test.shape[0])
    for i in range(n_models):
        y_pr_tr[indices_tr[i]] = predict_labels(weights[i], tX_tr_splitted[i])
        y_pr_te[indices_te[i]] = predict_labels(weights[i], tX_te_splitted[i])

    acc_tr = compute_accuracy(y_train, y_pr_tr)
    print(f"Total accuracy train: {acc_tr}")
    _, counts = np.unique(y_pr_te, return_counts=True)
    print(f"Distribution on test data class -1: {counts[0]}, class +1: {counts[1]}")

    create_csv_submission(ids_test, y_pr_te, OUTPUT_PATH)


if __name__ == "__main__":
    main()
