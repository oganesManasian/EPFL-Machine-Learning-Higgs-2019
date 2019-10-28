import numpy as np


def standardize(x):
    """Standardize an array of values."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    if abs(std_x) > 1e-10:
        x = x / std_x
    return x


def standardize_matrix(tX):
    """Standardize a matrix"""
    tX_standardized = np.empty_like(tX)
    for feature_index in range(tX_standardized.shape[1]):
        tX_standardized[:, feature_index] = standardize(tX[:, feature_index])
    return tX_standardized


def fill_missing_values(tX, filling_function=np.mean, mute=True):
    """Fill missing values using function defined by user (by default np.mean)"""
    missing_values_feature_indices = np.unique(np.vstack([np.argwhere(row == -999) for row in tX]))

    tX_filled = tX.copy()

    for feature_ind in missing_values_feature_indices:
        X = tX[:, feature_ind]
        filling_value = filling_function(X[X != -999])
        X[X == -999] = filling_value
        tX_filled[:, feature_ind] = X

        if not mute:
            print(f"Feature index {feature_ind}, filling_value {filling_value}")

    return tX_filled


def add_bias(tX):
    """Append column with ones"""
    return np.hstack((tX, np.ones((tX.shape[0], 1))))


def add_derived_feature(tX, feature_indices, deriving_function):
    """Appends columns which values are derived from columns with
    feature_indices indices using deriving_function function"""
    for feature_ind in feature_indices:
        tX = np.hstack((tX,
                        (deriving_function(tX[:, feature_ind])).reshape((tX.shape[0], 1))
                        ))
    return tX


def add_log(tX, feature_indices):
    """Appends columns which values are derived from columns with
    feature_indices indices applying logarithm"""
    return add_derived_feature(tX, feature_indices, deriving_function=lambda x: np.log(x - min(x) + 1))


def add_root(tX, feature_indices):
    """Appends columns which values are derived from columns with
    feature_indices indices applying square root"""
    return add_derived_feature(tX, feature_indices, deriving_function=lambda x: (x - min(x) + 1) ** (1 / 2))


def add_degree(tX, feature_indices, degree):
    """Appends columns which values are derived from columns with
    feature_indices indices applying power function with degree"""
    return add_derived_feature(tX, feature_indices, deriving_function=lambda x: x ** degree)


def preprocess_data(tX, y, params):
    """Preprocess data in the way stated in params argument"""
    description = params
    tX_new = tX.copy()

    if params.get("fill"):
        tX_new = fill_missing_values(tX_new)

    for deg in range(2, params.get("degree", 0) + 1):
        tX_new = add_degree(tX_new, range(0, tX.shape[1]), degree=deg)

    if params.get("log"):
        tX_new = add_log(tX_new, range(0, tX.shape[1]))

    if params.get("root"):
        tX_new = add_root(tX_new, range(0, tX.shape[1]))

    if params.get("bias"):
        tX_new = add_bias(tX_new)

    if params.get("standardize"):
        tX_new = standardize_matrix(tX_new)

    return tX_new, y, description


def divide_data(tX):
    """Divides dataset into 3 parts according to PRI_jet_num feature value.
    Undefined columns for particular values of PRI_jet_num are dropped"""
    cat_feature_ind = 22
    cat_feature_column = tX[:, cat_feature_ind]

    # Split dataset
    indices = [np.where(cat_feature_column == 0),
               np.where(cat_feature_column == 1),
               np.where(cat_feature_column >= 2)]

    tX_splitted = []

    # Drop features which are undefined with particular value of PRI_jet_num
    for i in range(len(indices)):
        tX_splitted.append(tX[indices[i]])

    features_to_drop = {0: [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28],
                        1: [4, 5, 6, 12, 22, 26, 27, 28],
                        2: [22]}

    for jet_class, feature_list in features_to_drop.items():
        features_to_leave = list(set(range(tX.shape[1])) - set(feature_list))
        tX_splitted[jet_class] = tX_splitted[jet_class][:, features_to_leave]

    return tX_splitted, indices


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
    y_undersampled = y[indices_to_choose]
    return tX_undersamled, y_undersampled


def find_outliers(X, mute=True):
    """Returns indices of outliers"""
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
