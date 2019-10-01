import numpy as np

"""In the above method signatures, for iterative methods, initial w is
the initial weight vector, gamma is the step-size, and max iters is the 
number of steps to run. lambda is always the regularization parameter. 
(Note that here we have used the trailing underscore because lambda is 
a reserved word in Python with a different meaning). For SGD, you must 
use the standard mini-batch-size 1 (sample just one datapoint)."""

"""Note that all functions should return: (w, loss), which is the last weight 
vector of the method, and the corresponding loss value (cost function)."""


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent"""
    raise NotImplementedError


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent"""
    raise NotImplementedError


def least_squares(y, tx):
    """Least squares regression using normal equations"""
    raise NotImplementedError


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""
    raise NotImplementedError


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD"""
    raise NotImplementedError


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD"""
    raise NotImplementedError
