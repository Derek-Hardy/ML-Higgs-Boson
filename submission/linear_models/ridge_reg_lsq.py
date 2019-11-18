import numpy as np

class RidgeRegressionLS:
    def __init__(self, lambda_=0.1, fit_intercept=True):
        self.theta = None
        self.lambda_ = lambda_
        self.fit_intercept = fit_intercept
    
    def __MSE(self, y, y_pred):
        return np.square(y - y_pred).mean() * 0.5

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        N, M = X.shape
        A = X.T.dot(X) + self.lambda_*2*N*np.eye(M)
        b = X.T.dot(y)
        self.theta = np.linalg.solve(A, b)
        return (self.theta, self.__MSE(y, X.dot(self.theta))) 
        
    def predict(self, X):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return np.dot(X, self.theta)
    