import numpy as np
from helpers import batch_iter

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
    return (1 / (2 * len(y))) * np.sum((y - np.dot(tX, w)) ** 2)


def compute_gradient(y, tX, w):
    """Compute the gradient."""
    return (-1 / len(y)) * np.dot(tX.T, y - np.dot(tX, w))


def least_squares_GD(y, tX, initial_w, max_iters, gamma):
    """Linear regression using gradient descent"""
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tX, w)
        loss = compute_loss(y, tX, w)
        w = w - gamma * gradient
        print("Gradient Descent({bi}/{ti}): loss={l}".format(
            bi=n_iter, ti=max_iters - 1, l=loss))

    return w, compute_loss(y, tX, w)


def least_squares_SGD(y, tX, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent"""
    w = initial_w
    for n_iter in range(max_iters):
        for y_new, x_new in batch_iter(y, tX, batch_size=1):
            gradient = compute_gradient(y_new, x_new, w)
            break
        loss = compute_loss(y, tX, w)
        w = w - gamma * gradient
        print("Gradient Descent({bi}/{ti}): loss={l}".format(
            bi=n_iter, ti=max_iters - 1, l=loss))

    return w, compute_loss(y, tX, w)


def least_squares(y, tX):
    """Least squares regression using normal equations"""
    w = np.dot(np.linalg.pinv(np.dot(tX.T, tX)), np.dot(tX.T, y))
    return w, compute_loss(y, tX, w)


def ridge_regression(y, tX, lambda_):
    """Ridge regression using normal equations"""
    w = np.linalg.solve(np.matmul(tX.T, tX) + lambda_ * np.identity(tX.shape[1]), np.matmul(tX.T, y))
    return w, compute_loss(y, tX, w)


def logistic_regression(y, tX, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD"""
    raise NotImplementedError


def reg_logistic_regression(y, tX, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD"""
    raise NotImplementedError
