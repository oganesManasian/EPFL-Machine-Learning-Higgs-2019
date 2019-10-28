import numpy as np
import matplotlib.pyplot as plt

from preprocessing_data import fill_missing_values, oversample, undersample, preprocess_data, divide_data
from helpers import split_data, predict_labels, compute_accuracy, load_csv_data, find_correlated_features, lambda_cv
import pandas as pd
from implementations import least_squares, ridge_regression

DATA_TRAIN_PATH = "../data/train.csv"
DATA_TEST_PATH = '../data/test.csv'
pd.set_option('display.max_colwidth', 100)


def solve(tX, y):
    tX_tr, y_tr, tX_te, y_te = split_data(tX, y, ratio=0.8, seed=2019)

    lambda_ = 1
    w, _ = ridge_regression(y_tr, tX_tr, lambda_)
    y_pr_tr = predict_labels(w, tX_tr)
    y_pr_te = predict_labels(w, tX_te)
    acc_tr = compute_accuracy(y_tr, y_pr_tr)
    acc_te = compute_accuracy(y_te, y_pr_te)

    return acc_tr, acc_te


preprocessing_options = [{"bias": False, "fill": False, "standardize": False, "degree": 1, "log": False, "root": False},
                         {"bias": True, "fill": False, "standardize": False, "degree": 1, "log": False, "root": False},

                         {"bias": True, "fill": True, "standardize": False, "degree": 1, "log": False, "root": False},
                         {"bias": True, "fill": True, "standardize": True, "degree": 1, "log": False, "root": False},

                         {"bias": True, "fill": True, "standardize": False, "degree": 2, "log": False, "root": False},
                         {"bias": True, "fill": True, "standardize": False, "degree": 2, "log": True, "root": True},
                         {"bias": True, "fill": True, "standardize": False, "degree": 3, "log": True, "root": True},
                         {"bias": True, "fill": True, "standardize": False, "degree": 4, "log": True, "root": True},
                         {"bias": True, "fill": True, "standardize": False, "degree": 5, "log": True, "root": True},

                         {"bias": True, "fill": True, "standardize": False, "degree": 6, "log": False, "root": False},
                         {"bias": True, "fill": True, "standardize": False, "degree": 6, "log": True, "root": False},
                         {"bias": True, "fill": True, "standardize": False, "degree": 6, "log": False, "root": True},
                         {"bias": True, "fill": True, "standardize": False, "degree": 6, "log": True, "root": True},
                         {"bias": True, "fill": True, "standardize": True, "degree": 6, "log": True, "root": True},

                         {"bias": True, "fill": True, "standardize": False, "degree": 7, "log": True, "root": True},
                         {"bias": True, "fill": True, "standardize": False, "degree": 8, "log": True, "root": True},
                         {"bias": True, "fill": True, "standardize": False, "degree": 9, "log": True, "root": True},
                         {"bias": True, "fill": True, "standardize": False, "degree": 10, "log": True, "root": True},
                         {"bias": True, "fill": True, "standardize": False, "degree": 11, "log": True, "root": True},

                         {"bias": True, "fill": True, "standardize": False, "degree": 12, "log": False, "root": False},
                         {"bias": True, "fill": True, "standardize": False, "degree": 12, "log": True, "root": False},
                         {"bias": True, "fill": True, "standardize": False, "degree": 12, "log": False, "root": True},
                         {"bias": True, "fill": True, "standardize": False, "degree": 12, "log": True, "root": True},
                         {"bias": True, "fill": True, "standardize": True, "degree": 12, "log": True, "root": True}]


def run_preprocessing_experiment(tX, y):
    results = pd.DataFrame(columns=["Preprocessing", "Accuracy train", "Accuracy test"])
    for preprocessing_param in preprocessing_options:
        tX_new, y_new, description = preprocess_data(tX, y, preprocessing_param)
        acc_tr, acc_te = solve(tX_new, y_new)

        print(preprocessing_param, f"Train: {acc_tr}, Test {acc_te}")
        results.loc[len(results)] = (description, acc_tr, acc_te)

    results = results.sort_values(["Accuracy test"], ascending=False)
    results.to_csv("Preprocessing experiment.csv", sep=";")


def balance_data(tX, y, method="oversample"):
    if method == "oversample":
        tX_new, y_new = oversample(tX, y)
        return tX_new, y_new, method
    elif method == "undersample":
        tX_new, y_new = undersample(tX, y)
        return tX_new, y_new, method
    elif method == "without balancing":
        return tX, y, method
    else:
        raise NotImplementedError


balancing_methods = ["without balancing", "oversample", "undersample"]


def run_balancing_experiment(tX, y):
    results = pd.DataFrame(columns=["Balancing", "Preprocessing", "Accuracy train", "Accuracy test"])

    for preprocessing_param in preprocessing_options:
        tX_prep, y_prep, desc_prep = preprocess_data(tX, y, preprocessing_param)
        for method in balancing_methods:
            tX_new, y_new, desc_balanc = balance_data(tX_prep, y_prep, method)
            acc_tr, acc_te = solve(tX_new, y_new)

            print(desc_balanc + " " + str(desc_prep), f"Train: {acc_tr}, Test {acc_te}")
            results.loc[len(results)] = (desc_balanc, desc_prep, acc_tr, acc_te)

    results = results.sort_values(["Accuracy test"], ascending=False)
    results.to_csv("Balancing experiment.csv", sep=";")


def fill_data(tX, y, method="mean"):
    if method == "mean":
        tX_new = fill_missing_values(tX, np.mean)
    elif method == "median":
        tX_new = fill_missing_values(tX, np.median)
    else:
        raise NotImplementedError
    return tX_new, y, method


filling_methods = ["mean", "median"]


def run_filling_experiment(tX, y):
    results = pd.DataFrame(columns=["Filling", "Preprocessing", "Accuracy train", "Accuracy test"])

    for preprocessing_param in preprocessing_options:
        tX_prep, y_prep, desc_prep = preprocess_data(tX, y, preprocessing_param)
        for method in filling_methods:
            tX_new, y_new, desc_fill = fill_data(tX_prep, y_prep, method)
            acc_tr, acc_te = solve(tX_new, y_new)

            print(desc_fill + " " + str(desc_prep), f"Train: {acc_tr}, Test {acc_te}")
            results.loc[len(results)] = (desc_fill, desc_prep, acc_tr, acc_te)

    results = results.sort_values(["Accuracy test"], ascending=False)
    results.to_csv("Filling experiment.csv", sep=";")


def experiment_for_submitting():
    y_train, tX_train, ids = load_csv_data(DATA_TRAIN_PATH)
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

    np.random.seed(2019)
    results = pd.DataFrame(columns=["Preprocessing", "Class -1 count", "Class +1 count"])

    for preprocessing_param in preprocessing_options:
        tX_stacked = np.vstack((tX_train, tX_test))
        prep_param = {"bias": True, "fill": True, "standardize": False, "degree": 11, "log": True, "root": True}
        tX_stacked_prep, _, desc_prep = preprocess_data(tX_stacked, None, prep_param)
        tX_train_prep, tX_test_prep = np.split(tX_stacked_prep, [len(tX_train)])

        lambda_ = lambda_cv(tX_train_prep, y_train)
        print(f"Best lambda: {lambda_}")
        w, _ = ridge_regression(y_train, tX_train_prep, lambda_)

        y_pred = predict_labels(w, tX_test_prep)
        uniq, count = np.unique(y_pred, return_counts=True)

        print(preprocessing_param, f"Class -1: {count[0]}, Class +1: {count[1]}")
        results.loc[len(results)] = (desc_prep, count[0], count[1])

    results.to_csv("Submitting experiment.csv", sep=";")


def feature_correlation_checking(tX, y, threshold):
    prep_param = {"bias": True, "fill": True, "standardize": False, "degree": 2, "log": False, "root": True}
    tX_new, y_new, _ = preprocess_data(tX, y, prep_param)

    correlated_features_pairs = find_correlated_features(tX_new, threshold)

    return correlated_features_pairs


def train_3models(tX, y):
    # Preprocess data together to have the same shifts while creating log or root features
    prep_param = {"bias": True, "fill": True, "standardize": False, "degree": 8, "log": True, "root": True}
    tX_new, y_new, _ = preprocess_data(tX, y, prep_param)

    tX_tr, y_tr, tX_te, y_te = split_data(tX_new, y_new, ratio=0.8, seed=2019)

    # Split data according to PRI_jet_num value
    tX_tr_splitted, indices_tr = divide_data(tX_tr)
    tX_te_splitted, indices_te = divide_data(tX_te)
    n_models = len(tX_tr_splitted)

    y_tr_splitted = []
    for i in range(len(indices_tr)):
        y_tr_splitted.append(y_tr[indices_tr[i]])
        print(tX_tr_splitted[i].shape)

    # Train
    weights = []
    for i in range(n_models):
        lambda_ = lambda_cv(tX_tr_splitted[i], y_tr_splitted[i])
        print(f"Class {i}, lambda: {lambda_}")
        weights.append(ridge_regression(y_tr_splitted[i], tX_tr_splitted[i], lambda_)[0])
        print(len(weights[-1]))

    # Predict
    y_pr_tr = np.zeros(y_tr.shape)
    y_pr_te = np.zeros(y_te.shape)
    for i in range(n_models):
        y_pr_tr[indices_tr[i]] = predict_labels(weights[i], tX_tr_splitted[i])
        y_pr_te[indices_te[i]] = predict_labels(weights[i], tX_te_splitted[i])

    # Get accuracy
    acc_tr = compute_accuracy(y_tr, y_pr_tr)
    acc_te = compute_accuracy(y_te, y_pr_te)
    print(f"Total accuracy tr: {acc_tr}, te: {acc_te}")

    for i in range(n_models):
        acc_tr = compute_accuracy(y_tr[indices_tr[i]], y_pr_tr[indices_tr[i]])
        acc_te = compute_accuracy(y_te[indices_te[i]], y_pr_te[indices_te[i]])
        print(f"Class {i}, Accuracy tr: {acc_tr}, te: {acc_te}")


def run_experiments():
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

    # run_preprocessing_experiment(tX, y)
    # run_balancing_experiment(tX, y)
    # run_filling_experiment(tX, y)
    # experiment_for_submitting()
    # feature_correlation_checking(tX, y)
    train_3models(tX, y)


if __name__ == "__main__":
    run_experiments()
