import numpy as np


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x


def standardize_matrix(tX):
    tX_standardized = np.empty_like(tX)
    for feature_index in range(tX_standardized.shape[1]):
        tX_standardized[:, feature_index] = standardize(tX[:, feature_index])
    return tX_standardized


def fill_missing_values(tX, filling_function=np.mean, mute=True):
    """Fill missing values using function defined by user (by default np.mean)"""

    tX_filled = tX.copy()

    missing_values_feature_indices = np.unique(np.vstack([np.argwhere(row == -999) for row in tX_filled]))

    for feature_ind in missing_values_feature_indices:
        X = tX_filled[:, feature_ind]
        filling_value = filling_function(X[X != -999])
        X[X == -999] = filling_value
        tX_filled[:, feature_ind] = X
        if not mute:
            print(f"Feature index {feature_ind}, filling_value {filling_value}")

    return tX_filled


def oversample(tX, y):
    """Oversample minor label to balance classes"""
    unique, count = np.unique(y, return_counts=True)
    samples_number_to_add = max(count) - min(count)
    indices_minor_class = np.where(y == 1)[0]
    indices_to_add = np.random.choice(indices_minor_class, samples_number_to_add)
    tX_oversamled = np.vstack((tX, tX[indices_to_add]))
    y_oversampled = np.hstack((y, y[indices_to_add]))
    return tX_oversamled, y_oversampled


def undersample(tX, y):
    """Undersample major label to balance classes"""
    unique, count = np.unique(y, return_counts=True)
    samples_number_to_choose = min(count)
    indices_major_class = np.where(y == -1)[0]
    indices_minor_class = np.where(y == 1)[0]
    indices_to_choose = np.random.choice(indices_major_class, samples_number_to_choose)
    indices_to_choose = np.hstack((indices_to_choose, indices_minor_class))
    tX_undersamled = tX[indices_to_choose]
    y_undersampled = tX[indices_to_choose]
    return tX_undersamled, y_undersampled


def find_outliers(X, mute=True):
    """returns indices of outliers"""
    LQ = np.quantile(X, q=0.25)
    HQ = np.quantile(X, q=0.75)
    IQR = HQ - LQ
    if not mute:
        print("Boundaries: (", LQ - 1.5 * IQR, ",", HQ + 1.5 * IQR, ")")
    outliers_indices = np.where((X < LQ - 1.5 * IQR) | (X > HQ + 1.5 * IQR))[0]
    return outliers_indices


def drop_outliers(tX, y, mute=True):
    """Drop outliers (First fill missing values)"""
    outlier_indices = set()

    missing_values_feature_indices = np.unique(np.vstack([np.argwhere(row == -999) for row in tX]))
    columns_without_missing_data = list(set(range(tX.shape[1])) - set(np.unique(missing_values_feature_indices)))

    for feature_ind in columns_without_missing_data:  # All columns if missing values are filled
        indices = find_outliers(tX[:, feature_ind], mute)
        if not mute:
            print(f"Feature index {feature_ind}, {len(indices)} outliers")
        outlier_indices = outlier_indices | set(indices)

    if not mute:
        print("Total number of outliers", len(outlier_indices))

    indices_without_outliers = list(set(range(tX.shape[0])) - outlier_indices)
    tX_without_outliers = tX[indices_without_outliers]
    y_without_outliers = y[indices_without_outliers]
    return tX_without_outliers, y_without_outliers
