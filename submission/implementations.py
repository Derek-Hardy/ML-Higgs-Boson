from linear_models.linear_reg_gd import *
from linear_models.linear_reg_lsq import *
from linear_models.logistic_reg import *
from linear_models.ridge_reg_lsq import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    model = LinearRegressionGD(lambda_=0, fit_intercept=False)
    return model.fit(tx, y, step=gamma, max_iter=max_iters, initial_w=initial_w)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    model = LinearRegressionGD(lambda_=0, fit_intercept=False)
    return model.fit(tx, y, step=gamma, max_iter=max_iters, batch_size=1, initial_w=initial_w)

def least_squares(y, tx):
    model = LeastSquares(fit_intercept=False)
    return model.fit(tx, y)

def ridge_regression(y, tx, lambda_):
    model = RidgeRegressionLS(lambda_=lambda_, fit_intercept=False)
    return model.fit(tx, y)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    # flatten to ensure algorithm works e.g need (N,) instead of (N,1)
    if len(y.shape) > 1:
        y = y.flatten()

    model = LogisticRegression(fit_intercept=False)
    return model.fit(tx, y, step=gamma, max_iter=max_iters, initial_w=initial_w)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    # flatten to ensure algorithm works e.g need (N,) instead of (N,1)
    if len(y.shape) > 1:
        y = y.flatten()

    model = LogisticRegression(lambda_=lambda_, fit_intercept=False)
    return model.fit(tx, y, step=gamma, max_iter=max_iters, initial_w=initial_w) 
