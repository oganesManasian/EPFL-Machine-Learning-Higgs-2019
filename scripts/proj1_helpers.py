# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np

feature_names = ['DER_mass_MMC',
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
                 'PRI_jet_all_pt']


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


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred


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
