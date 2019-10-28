# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
from __future__ import print_function

import csv
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
import ipywidgets as widgets

init_feature_names = np.array(['DER_mass_MMC',
                               'DER_mass_transverse_met_lep',
                               'DER_mass_vis',
                               'DER_pt_h',
                               'DER_deltaeta_jet_jet',
                               'DER_mass_jet_jet',
                               'DER_prodeta_jet_jet',
                               'DER_deltar_tau_lep',
                               'DER_pt_tot',
                               'DER_sum_pt',
                               'DER_pt_ratio_lep_tau',
                               'DER_met_phi_centrality',
                               'DER_lep_eta_centrality',
                               'PRI_tau_pt',
                               'PRI_tau_eta',
                               'PRI_tau_phi',
                               'PRI_lep_pt',
                               'PRI_lep_eta',
                               'PRI_lep_phi',
                               'PRI_met',
                               'PRI_met_phi',
                               'PRI_met_sumet',
                               'PRI_jet_num',
                               'PRI_jet_leading_pt',
                               'PRI_jet_leading_eta',
                               'PRI_jet_leading_phi',
                               'PRI_jet_subleading_pt',
                               'PRI_jet_subleading_eta',
                               'PRI_jet_subleading_phi',
                               'PRI_jet_all_pt'])


def interact_feature_hist(tX, y, cur_feature_names):
    def plot_feature_hist(feature_ind, drop_nan):
        indices_class_pos = np.where(y == 1)
        indices_class_neg = np.where(y == -1)

        tX_pos = tX[indices_class_pos, feature_ind]
        tX_neg = tX[indices_class_neg, feature_ind]

        if drop_nan:
            tX_pos = tX_pos[tX_pos != -999]
            tX_neg = tX_neg[tX_neg != -999]

        plt.figure(figsize=(15, 6), dpi=80)
        plt.subplot(1, 2, 1)
        plt.title(cur_feature_names[feature_ind] + f" (Feature #{feature_ind})")
        plt.hist(tX_pos.T, bins=100, color="r", label="+1")
        plt.hist(tX_neg.T, bins=100, color="b", label="-1")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.title(cur_feature_names[feature_ind] + f" (Feature #{feature_ind})")
        plt.hist(tX_neg.T, bins=100, color="b", label="-1")
        plt.hist(tX_pos.T, bins=100, color="r", label="+1")
        plt.legend()

    interact(plot_feature_hist,
             feature_ind=widgets.BoundedIntText(value=0, min=0, max=tX.shape[1] - 1, step=1,
                                                description='Feature index:', disabled=False),
             drop_nan=widgets.Checkbox(value=False, description='Drop missing values', disabled=False)
             )
    plt.show()


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def compute_column_correlation(tX):
    """Compute correlation of columns in matrix"""
    n_features = tX.shape[1]
    corr_matrix = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(i, n_features):
            corr_matrix[i, j] = corr_matrix[j, i] = np.corrcoef(tX[:, i], tX[:, j])[0, 1]

    return corr_matrix


def find_correlated_features(tX, threshold):
    """Find correlated features using correlation matrix"""
    corr_matrix = compute_column_correlation(tX)
    print(f"{np.count_nonzero(np.isnan(corr_matrix))} nan elements")

    # Extract pairs of correlated features
    correlated_features = np.where(corr_matrix >= threshold)
    correlated_features_pairs = [(i, j) for (i, j) in zip(correlated_features[0], correlated_features[1]) if i < j]

    return correlated_features_pairs


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def split_data(x, y, ratio, seed=1):
    """
    Splits the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    """
    np.random.seed(seed)
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation]
    y = y[permutation]
    n = int(ratio * x.shape[0])
    x_tr = x[:n]
    x_te = x[n:]
    y_tr = y[:n]
    y_te = y[n:]
    return x_tr, y_tr, x_te, y_te


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred


def compute_accuracy(y_true, y_pred):
    """Gets accuracy by computing percent of equal elements in two arrays"""
    return np.round(np.sum(y_pred * y_true > 0) / len(y_true), 6)


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})
