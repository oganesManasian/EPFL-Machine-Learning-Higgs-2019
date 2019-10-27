import numpy as np
import matplotlib.pyplot as plt

from preprocessing_data import add_bias, add_degree, add_root, add_log, fill_missing_values, standardize_matrix, \
    oversample, undersample
from helpers import split_data, predict_labels, compute_accuracy, load_csv_data
import pandas as pd
from implementations import least_squares, ridge_regression

DATA_TRAIN_PATH = "../data/train.csv"
pd.set_option('display.max_colwidth', 100)


def compare_solvers(tX, y):
    tX_tr, y_tr, tX_te, y_te = split_data(tX, y, ratio=0.8, seed=2019)
    accs_tr = []
    accs_te = []

    w, _ = least_squares(y_tr, tX_tr)
    y_pr_tr = predict_labels(w, tX_tr)
    y_pr_te = predict_labels(w, tX_te)
    accs_tr.append(compute_accuracy(y_tr, y_pr_tr))
    accs_te.append(compute_accuracy(y_te, y_pr_te))

    lambda_ = 1
    w, _ = ridge_regression(y_tr, tX_tr, lambda_)
    y_pr_tr = predict_labels(w, tX_tr)
    y_pr_te = predict_labels(w, tX_te)
    accs_tr.append(compute_accuracy(y_tr, y_pr_tr))
    accs_te.append(compute_accuracy(y_te, y_pr_te))

    return max(accs_te)


def preprocessing_option_0(tX, y):
    description = "Raw data"
    return tX, y, description


def preprocessing_option_1(tX, y):
    description = "Bias"
    tX_new = tX.copy()

    add_bias(tX_new)
    return tX_new, y, description


def preprocessing_option_2(tX, y):
    description = "Bias + filled missing values"
    tX_new = tX.copy()

    tX_new = add_bias(tX_new)
    tX_new = fill_missing_values(tX_new)
    return tX_new, y, description


def preprocessing_option_3(tX, y):
    description = "Bias + filled missing values + standardizing"
    tX_new = tX.copy()

    tX_new = add_bias(tX_new)
    tX_new = fill_missing_values(tX_new)
    tX_new = standardize_matrix(tX_new)

    return tX_new, y, description


def preprocessing_option_4(tX, y):
    description = "Bias + filled missing values + 2 degree"
    tX_new = tX.copy()

    tX_new = add_bias(tX_new)
    tX_new = fill_missing_values(tX_new)
    tX_new = add_degree(tX_new, range(0, tX.shape[1]), degree=2)

    return tX_new, y, description


def preprocessing_option_5(tX, y):
    description = "Bias + filled missing values + 2, 3 degree"
    tX_new = tX.copy()

    tX_new = add_bias(tX_new)
    tX_new = fill_missing_values(tX_new)
    tX_new = add_degree(tX_new, range(0, tX.shape[1]), degree=2)
    tX_new = add_degree(tX_new, range(0, tX.shape[1]), degree=3)

    return tX_new, y, description


def preprocessing_option_6(tX, y):
    description = "Bias + filled missing values + 2 degree + square root"
    tX_new = tX.copy()

    tX_new = add_bias(tX_new)
    tX_new = fill_missing_values(tX_new)
    tX_new = add_degree(tX_new, range(0, tX.shape[1]), degree=2)
    tX_new = add_root(tX_new, range(0, tX.shape[1]))

    return tX_new, y, description


def preprocessing_option_7(tX, y):
    description = "Bias + filled missing values + 2 degree + log"
    tX_new = tX.copy()

    tX_new = add_bias(tX_new)
    tX_new = fill_missing_values(tX_new)
    tX_new = add_degree(tX_new, range(0, tX.shape[1]), degree=2)
    tX_new = add_log(tX_new, range(0, tX.shape[1]))

    return tX_new, y, description


def preprocessing_option_8(tX, y):
    description = "Bias + filled missing values + 2, 3 degree + root"
    tX_new = tX.copy()

    tX_new = add_bias(tX_new)
    tX_new = fill_missing_values(tX_new)
    tX_new = add_degree(tX_new, range(0, tX.shape[1]), degree=2)
    tX_new = add_degree(tX_new, range(0, tX.shape[1]), degree=3)
    tX_new = add_root(tX_new, range(0, tX.shape[1]))

    return tX_new, y, description


def preprocessing_option_9(tX, y):
    description = "Bias + filled missing values + 2, 3 degree + log"
    tX_new = tX.copy()

    tX_new = add_bias(tX_new)
    tX_new = fill_missing_values(tX_new)
    tX_new = add_degree(tX_new, range(0, tX.shape[1]), degree=2)
    tX_new = add_degree(tX_new, range(0, tX.shape[1]), degree=3)
    tX_new = add_log(tX_new, range(0, tX.shape[1]))

    return tX_new, y, description


def preprocessing_option_10(tX, y):
    description = "Bias + filled missing values + 2 degree + log + root"
    tX_new = tX.copy()

    tX_new = add_bias(tX_new)
    tX_new = fill_missing_values(tX_new)
    tX_new = add_degree(tX_new, range(0, tX.shape[1]), degree=2)
    tX_new = add_log(tX_new, range(0, tX.shape[1]))
    tX_new = add_root(tX_new, range(0, tX.shape[1]))

    return tX_new, y, description


def preprocessing_option_11(tX, y):
    description = "Bias + filled missing values + 2, 3 degree + log + root"
    tX_new = tX.copy()

    tX_new = add_bias(tX_new)
    tX_new = fill_missing_values(tX_new)
    tX_new = add_degree(tX_new, range(0, tX.shape[1]), degree=2)
    tX_new = add_degree(tX_new, range(0, tX.shape[1]), degree=3)
    tX_new = add_log(tX_new, range(0, tX.shape[1]))
    tX_new = add_root(tX_new, range(0, tX.shape[1]))

    return tX_new, y, description


preprocessing_options = [preprocessing_option_0, preprocessing_option_1, preprocessing_option_2,
                         preprocessing_option_3, preprocessing_option_4, preprocessing_option_5,
                         preprocessing_option_6, preprocessing_option_7, preprocessing_option_8,
                         preprocessing_option_9, preprocessing_option_10, preprocessing_option_11]


def run_preprocessing_experiment(tX, y):
    results = pd.DataFrame(columns=["Preprocessing", "Accuracy"])
    for preprocessing_option in preprocessing_options:
        tX_new, y_new, description = preprocessing_option(tX, y)
        best_acc = compare_solvers(tX_new, y_new)

        results.loc[len(results)] = (description, best_acc)

    results = results.sort_values(["Accuracy"], ascending=False)
    results.to_csv("Preprocessing experiment.csv", sep=";")



def balancing_option_0(tX, y):
    description = "Without balancing"
    return tX, y, description


def balancing_option_1(tX, y):
    description = "Oversample"
    tX_new, y_new = oversample(tX, y)
    return tX_new, y_new, description


def balancing_option_2(tX, y):
    description = "Undersample"
    tX_new, y_new = undersample(tX, y)
    return tX_new, y_new, description


balancing_options = [balancing_option_0, balancing_option_1, balancing_option_2]


def run_balancing_experiment(tX, y):
    results = pd.DataFrame(columns=["Balancing", "Preprocessing", "Accuracy"])

    for preprocessing_option in preprocessing_options:
        tX_prep, y_prep, desc_prep = preprocessing_option(tX, y)
        for balancing_option in balancing_options:
            tX_new, y_new, desc_balanc = balancing_option(tX_prep, y_prep)
            best_acc = compare_solvers(tX_new, y_new)

            results.loc[len(results)] = (desc_balanc, desc_prep, best_acc)

    results = results.sort_values(["Accuracy"], ascending=False)
    results.to_csv("Balancing experiment.csv", sep=";")


def filling_option_1(tX, y):
    description = "Mean"
    tX_new = fill_missing_values(tX, np.mean)
    return tX_new, y, description


def filling_option_2(tX, y):
    description = "Median"
    tX_new = fill_missing_values(tX, np.median)
    return tX_new, y, description


filling_options = [filling_option_1, filling_option_2]


def run_filling_experiment(tX, y):
    results = pd.DataFrame(columns=["Filling", "Preprocessing", "Accuracy"])

    for preprocessing_option in preprocessing_options:
        tX_prep, y_prep, desc_prep = preprocessing_option(tX, y)
        for filling_option in filling_options:
            tX_new, y_new, desc_fill = filling_option(tX_prep, y_prep)
            best_acc = compare_solvers(tX_new, y_new)

            results.loc[len(results)] = (desc_fill, desc_prep, best_acc)

    results = results.sort_values(["Accuracy"], ascending=False)
    results.to_csv("Filling experiment.csv", sep=";")


def lambda_cv(tX, y, plot=False):
    lambdas = np.logspace(-5, 5, 15)

    tX_new, y_new, _ = preprocessing_option_5(tX, y)
    tX_tr, y_tr, tX_te, y_te = split_data(tX_new, y_new, ratio=0.8, seed=1)

    accs_tr = []
    accs_te = []

    for lambda_ in lambdas:
        w, _ = ridge_regression(y_tr, tX_tr, lambda_)
        y_pr_tr = predict_labels(w, tX_tr)
        y_pr_te = predict_labels(w, tX_te)
        accs_tr.append(compute_accuracy(y_tr, y_pr_tr))
        accs_te.append(compute_accuracy(y_te, y_pr_te))

    min_acc = max(accs_te)
    best_lambda = lambdas[np.argwhere(accs_te == min_acc)][0]

    if plot:
        plt.plot(lambdas, accs_tr, label="Train")
        plt.plot(lambdas, accs_te, label="Test")
        plt.plot(best_lambda, min_acc, "*", label="Best value")
        plt.xlabel("Lambda")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

    return best_lambda, min_acc


def experiment_for_submitting(tX, y):
    DATA_TEST_PATH = '../data/test.csv'

    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

    np.random.seed(2019)
    results = pd.DataFrame(columns=["Preprocessing", "Class -1 count", "Class +1 count"])
    for preprocessing_option in preprocessing_options:
        tX_prep, _, desc_prep = preprocessing_option(tX, None)
        tX_test_prep, *_ = preprocessing_option(tX_test, None)

        lambda_ = 1
        w, _ = ridge_regression(y, tX_prep, lambda_)

        y_pred = predict_labels(w, tX_test_prep)
        uniq, count = np.unique(y_pred, return_counts=True)
        results.loc[len(results)] = (desc_prep, count[0], count[1])

    results.to_csv("Submitting experiment.csv", sep=";")


def run_experiments():
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

    # run_preprocessing_experiment(tX, y)
    # run_balancing_experiment(tX, y)
    # run_filling_experiment(tX, y)
    # print("Best lambda:", lambda_cv(tX, y, plot=True)[0])
    # experiment_for_submitting(tX, y)


if __name__ == "__main__":
    run_experiments()
