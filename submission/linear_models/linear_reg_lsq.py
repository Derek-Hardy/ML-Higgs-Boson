import numpy as np

class LeastSquares:
    def __init__(self, fit_intercept=True):
        self.theta = None
        self.fit_intercept = fit_intercept
    
    def __MSE(self, y, y_pred):
        return np.square(y - y_pred).mean() * 0.5

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        self.theta = np.linalg.solve(X.T.dot(X), X.T.dot(y))
        return (self.theta, self.__MSE(y, X.dot(self.theta)))
        
    def predict(self, X):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return np.dot(X, self.theta)
