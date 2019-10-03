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

def compute_loss(y, tx, w):
    """Compute the loss."""
    return (1/2*len(y))*np.sum((y-np.dot(tx,w))**2)


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    return (-1/len(y))*np.dot(tx.T,y-np.dot(tx,w))


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent"""
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y,tx,w)
        loss = compute_loss(y,tx,w)
        w = w - gamma*gradient
        print("Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))

    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent"""
    w = initial_w
    for n_iter in range(max_iters):
        for y_new,x_new in batch_iter(y,tx,batch_size=1):
            gradient = compute_gradient(y_new,x_new,w)
            break
        loss = compute_loss(y_new,x_new,w)
        w = w - gamma*gradient
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
        
    return w, loss


def least_squares(y, tx):
    """Least squares regression using normal equations"""
    return np.dot(np.linalg.pinv(np.dot(tx.T, tx)), np.dot(tx.T, y))


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""
    raise NotImplementedError


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD"""
    raise NotImplementedError


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD"""
    raise NotImplementedError
