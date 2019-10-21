import numpy as np
from proj1_helpers import batch_iter

"""In the above method signatures, for iterative methods, initial w is
the initial weight vector, gamma is the step-size, and max iters is the
number of steps to run. lambda is always the regularization parameter.
(Note that here we have used the trailing underscore because lambda is
a reserved word in Python with a different meaning). For SGD, you must
use the standard mini-batch-size 1 (sample just one datapoint)."""

"""Note that all functions should return: (w, loss), which is the last weight
vector of the method, and the corresponding loss value (cost function)."""


def compute_loss(y, tX, w):
    """Compute the loss."""
    return np.round((1 / (2 * len(y))) * np.sum((y - np.dot(tX, w)) ** 2), 4)


def compute_gradient(y, tX, w):
    """Compute the gradient."""
    return (-1 / len(y)) * np.dot(tX.T, y - np.dot(tX, w))


def least_squares_GD(y, tX, initial_w, max_iters, gamma, mute=True):
    """Linear regression using gradient descent"""
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tX, w)
        loss = compute_loss(y, tX, w)
        w = w - gamma * gradient
        if not mute:
            print("Gradient Descent({bi}/{ti}): loss={l}".format(
                bi=n_iter, ti=max_iters - 1, l=loss))

    return w, compute_loss(y, tX, w)


def least_squares_SGD(y, tX, initial_w, max_iters, gamma, mute=True):
    """Linear regression using stochastic gradient descent"""
    w = initial_w
    for n_iter in range(max_iters):
        for y_new, x_new in batch_iter(y, tX, batch_size=1):
            gradient = compute_gradient(y_new, x_new, w)
            break
        loss = compute_loss(y, tX, w)
        w = w - gamma * gradient
        if not mute:
            print("Stochastic Gradient Descent({bi}/{ti}): loss={l}".format(
                bi=n_iter, ti=max_iters - 1, l=loss))

    return w, compute_loss(y, tX, w)


def least_squares(y, tX):
    """Least squares regression using normal equations"""
    w = np.linalg.solve(np.matmul(tX.T, tX), np.matmul(tX.T, y))
    return w, compute_loss(y, tX, w)


def ridge_regression(y, tX, lambda_):
    """Ridge regression using normal equations"""
    w = np.linalg.solve(np.matmul(tX.T, tX) + lambda_ * np.identity(tX.shape[1]), np.matmul(tX.T, y))
    return w, compute_loss(y, tX, w)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_log_loss(y, tX, w):
    n = y.size
    prob = sigmoid(tX.dot(w))
    return -np.mean((y * np.log(prob) + (1 - y) * np.log(1 - prob)))


def compute_log_gradient(y, tX, w):
    n = y.size
    return (1 / n) * tX.T.dot(sigmoid(tX.dot(w)) - y)


def logistic_regression(y, tX, initial_w, max_iters, gamma, mute=True):
    """Logistic regression using gradient descent or SGD"""
    w = initial_w
    for n_iter in range(max_iters):
        for y_new, x_new in batch_iter(y, tX, batch_size=1):
            gradient = compute_log_gradient(y_new, x_new, w)
        loss = compute_log_loss(y, tX, w)
        w = w - gamma * gradient
        if not mute:
            print("Stochastic Gradient Descent({bi}/{ti}): loss={l}".format(
                bi=n_iter, ti=max_iters - 1, l=loss))
    return w, compute_log_loss(y, tX, w)


def compute_reg_log_loss(y, tX, w, lambda_):
    n = y.size
    reg_term = (lambda_ / (2 * n) * sum(w ** 2))
    return compute_log_loss(y, tX, w) + reg_term


def compute_reg_log_gradient(y, tX, w, lambda_):
    reg_term = lambda_ / y.size * sum(w)
    return compute_log_gradient(y, tX, w) + reg_term


def reg_logistic_regression(y, tX, lambda_, initial_w, max_iters, gamma, mute=True):
    """Regularized logistic regression using gradient descent or SGD"""
    w = initial_w
    for n_iter in range(max_iters):
        for y_new, x_new in batch_iter(y, tX, batch_size=64):
            gradient = compute_reg_log_gradient(y_new, x_new, w, lambda_)
        loss = compute_reg_log_loss(y, tX, w, lambda_)
        w = w - gamma * gradient
        if not mute:
            print("Stochastic Gradient Descent({bi}/{ti}): loss={l}".format(
                bi=n_iter, ti=max_iters - 1, l=loss))
    return w, compute_reg_log_loss(y, tX, w, lambda_)
