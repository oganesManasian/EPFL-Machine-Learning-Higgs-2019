import numpy as np
from helpers import batch_iter
from numpy.linalg import LinAlgError


def compute_loss(y, tX, w):
    """Compute loss using Mean Squared Error.

    Parameters
    ----------
    y: ndarray, shape (n,)
        Output vector.

    tX: ndarray, shape (n, d)
        Training data.

    w: ndarray, shape (d,)
        Weight vector.

    Returns
    -------
    loss : ndarray, shape (n,)
        Mean Squared Error vector.
    """
    n = y.size
    e = y - tX.dot(w)
    loss = 1/(2 * n) * (e.T.dot(e))
    return loss

def compute_gradient(y, tX, w):
    """Compute the gradient of Mean Squared Error.

    Parameters
    ----------
    y: ndarray, shape (n,)
        Output vector.

    tX: ndarray, shape (n, d)
        Training data.

    w: ndarray, shape (d,)
        Weight vector.

    Returns
    -------
    gradient : ndarray, shape (n,)
        Gradient vector of Mean Squared Error.
    """
    n = y.size
    e = y - tX.dot(w)
    gradient = -1/n * tX.T.dot(e)
    return gradient


def least_squares_GD(y, tX, initial_w, max_iters, gamma, mute=True):
    """Linear regression using gradient descent.

    Parameters
    ----------
    y: ndarray, shape (n,)
        Output vector.

    tX: ndarray, shape (n, d)
        Training data.

    initial_w: ndarray, shape (d,)
        Initial weight vector.

    max_iters: int
        Maximum number of iterations.

    gamma: float
        Gradient step size.

    mute: bool, default True
        Print iterations information.

    Returns
    -------
    w: ndarray, shape (d,)
        Weight vector.

    final_loss : ndarray, shape (n,)
        Mean Squared Error vector.
    """
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tX, w)
        loss = compute_loss(y, tX, w)
        w = w - gamma * gradient
        if not mute:
            print("Gradient Descent({bi}/{ti}): loss={l}".format(
                bi=n_iter, ti=max_iters - 1, l=loss))

    final_loss = compute_loss(y, tX, w)
    return w, final_loss


def least_squares_SGD(y, tX, initial_w, max_iters, gamma, mute=True):
    """Linear regression using stochastic gradient descent.

    Parameters
    ----------
    y: ndarray, shape (n,)
        Output vector.

    tX: ndarray, shape (n, d)
        Training data.

    initial_w: ndarray, shape (d,)
        Initial weight vector.

    max_iters: int
        Maximum number of iterations.

    gamma: float
        Gradient step size.

    mute: bool, default True
        Print iterations information.

    Returns
    -------
    w: ndarray, shape (d,)
        Weight vector.

    final_loss : ndarray, shape (n,)
        Mean Squared Error vector.
    """
    w = initial_w
    for n_iter in range(max_iters):
        for y_new, x_new in batch_iter(y, tX, batch_size=1):
            gradient = compute_gradient(y_new, x_new, w)
        loss = compute_loss(y, tX, w)
        w = w - gamma * gradient
        if not mute:
            print("Stochastic Gradient Descent({bi}/{ti}): loss={l}"
                .format(bi=n_iter, ti=max_iters - 1, l=loss))

    final_loss = compute_loss(y, tX, w)
    return w, final_loss


def least_squares(y, tX):
    """Least squares regression using normal equations

    Parameters
    ----------
    y: ndarray, shape (n,)
        Output vector.

    tX: ndarray, shape (n, d)
        Training data.

    Returns
    -------
    w: ndarray, shape (d,)
        Weight vector.

    loss : ndarray, shape (n,)
        Mean Squared Error vector.
    """
    try:
        w = np.linalg.solve(tX.T.dot(tX), tX.T.dot(y))
    except LinAlgError:
        print("Singular matrix exception")
        w = np.dot(np.linalg.pinv(tX.T.dot(tX)), tX.T.dot(y))
    loss = compute_loss(y, tX, w)
    return w, loss


def ridge_regression(y, tX, lambda_):
    """Ridge regression using normal equations

    Parameters
    ----------
    y: ndarray, shape (n,)
        Output vector.

    tX: ndarray, shape (n, d)
        Training data.

    Returns
    -------
    w: ndarray, shape (d,)
        Weight vector.

    loss : ndarray, shape (n,)
        Mean Squared Error vector.
    """
    d = tX.shape[1]
    try:
        w = np.linalg.solve(tX.T.dot(tX) + lambda_ * np.identity(d), tX.T.dot(y))
    except LinAlgError:
        print("Singular matrix exception")
        w = np.dot(np.linalg.pinv(tX.T.dot(tX) + lambda_ * np.identity(d), tX.T.dot(y))
    loss = compute_loss(y, tX, w)
    return w, loss


def sigmoid(z):
    """Computes sigmoid function.

                       1
    sigmoid(x) =  ------------
                   1 + e^(-z)

    Parameters
    ----------
    z : float
        Real-value argument.

    Returns
    -------
    s : float
        Sigmoid function.
    """
    if z>=0:
        return 1 / (1 + np.exp(-z))
    else:
        return np.exp(z) / (1 + np.exp(z))


def compute_log_loss(y, tX, w):
    """Compute logarithmic loss.

    Parameters
    ----------
    y: ndarray, shape (n,)
        Output vector.

    tX: ndarray, shape (n, d)
        Training data.

    w: ndarray, shape (d,)
        Weight vector.

    Returns
    -------
    loss: ndarray, shape (d,)
        Weight vector.
    """
    prob = sigmoid(tX.dot(w))
    prob_clip = np.clip(prob, 1e-15, 1 - 1e-15)  # Use this trick to avoid numerical errors
    loss = -np.mean((y * np.log(prob_clip) + (1 - y) * np.log(1 - prob_clip)))
    return loss


def compute_log_gradient(y, tX, w):
    """Compute logarithmic gradient.

    Parameters
    ----------
    y: ndarray, shape (n,)
        Output vector.

    tX: ndarray, shape (n, d)
        Training data.

    w: ndarray, shape (d,)
        Weight vector.

    Returns
    -------
    gradient : ndarray, shape (n,)
        Gradient vector.
    """
    n = y.size
    gradient = (1 / n) * tX.T.dot(sigmoid(tX.dot(w)) - y)
    return gradient


def logistic_regression(y, tX, initial_w, max_iters, gamma, mute=True):
    """Logistic regression using gradient descent.

    Parameters
    ----------
    y: ndarray, shape (n,)
        Output vector.

    tX: ndarray, shape (n, d)
        Training data.

    initial_w: ndarray, shape (d,)
        Initial weight vector.

    max_iters: int
        Maximum number of iterations.

    gamma: float
        Gradient step size.

    mute: bool, default True
        Print iterations information.

    Returns
    -------
    w: ndarray, shape (d,)
        Weight vector.

    final_loss : ndarray, shape (n,)
        Loss vector.
    """
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_log_gradient(y, tX, w)
        loss = compute_log_loss(y, tX, w)
        w = w - gamma * gradient
        if not mute:
            print("Gradient Descent({bi}/{ti}): loss={l}"
            .format(bi=n_iter, ti=max_iters - 1, l=loss))

    final_loss = compute_log_loss(y, tX, w)
    return w, final_loss


def compute_reg_log_loss(y, tX, w, lambda_):
    """Computes regularized logarithmic loss.

    Parameters
    ----------
    y: ndarray, shape (n,)
        Output vector.

    tX: ndarray, shape (n, d)
        Training data.

    w: ndarray, shape (d,)
        Weight vector.

    Returns
    -------
    loss: ndarray, shape (d,)
        Weight vector.
    """
    n = y.size
    reg_term = (lambda_ / (2 * n) * np.sum(w ** 2))
    loss = compute_log_loss(y, tX, w) + reg_term
    return loss


def compute_reg_log_gradient(y, tX, w, lambda_):
    """Compute regularized logarithmic gradient.

    Parameters
    ----------
    y: ndarray, shape (n,)
        Output vector.

    tX: ndarray, shape (n, d)
        Training data.

    w: ndarray, shape (d,)
        Weight vector.

    Returns
    -------
    gradient : ndarray, shape (n,)
        Gradient vector.
    """
    reg_term = lambda_ / y.size * sum(w)
    gradient = compute_log_gradient(y, tX, w) + reg_term
    return gradient


def reg_logistic_regression(y, tX, lambda_, initial_w, max_iters, gamma, mute=True):
    """Regularized logistic regression using gradient descent.

    Parameters
    ----------
    y: ndarray, shape (n,)
        Output vector.

    tX: ndarray, shape (n, d)
        Training data.

    initial_w: ndarray, shape (d,)
        Initial weight vector.

    max_iters: int
        Maximum number of iterations.

    gamma: float
        Gradient step size.

    mute: bool, default True
        Print iterations information.

    Returns
    -------
    w: ndarray, shape (d,)
        Weight vector.

    final_loss : ndarray, shape (n,)
        Loss vector.
    """
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_reg_log_gradient(y, tX, w, lambda_)
        loss = compute_reg_log_loss(y, tX, w, lambda_)
        w = w - gamma * gradient
        if not mute:
            print("Regularized logistic regression using Gradient Descent({bi}/{ti}): loss={l}"
            .format(bi=n_iter, ti=max_iters - 1, l=loss))

    final_loss = compute_reg_log_loss(y, tX, w, lambda_)
    return w, final_loss
