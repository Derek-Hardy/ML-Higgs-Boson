import numpy as np
from proj1_helpers import *

class LinearRegressionGD:
    def __init__(self, lambda_=0.1, fit_intercept=True):
        self.lambda_ = lambda_
        self.fit_intercept = fit_intercept
    
    def fit(self, X, y, step=0.5, threshold=1e-8, max_iter=100, batch_size=-1, initial_w=0):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        loss_prev = np.inf
        self.theta = np.zeros(X.shape[1]).reshape(-1, 1)
        if (np.any(initial_w)):
            self.theta = initial_w

        it = None
        if batch_size != -1:
            it = batch_iter(y, X, batch_size, num_batches=max_iter)
        
        for _ in range(max_iter):            
            y_pred = np.dot(X, self.theta)
            loss = self._MSE(y, y_pred)
            if loss_prev - loss < threshold and batch_size == -1:
                break
            loss_prev = loss
            
            if batch_size != -1:
                y_batch, X_batch = next(it)
                y_pred_batch = np.dot(X_batch, self.theta)
                self.theta -= step*self._MSE_Gradient(X_batch, y_batch, y_pred_batch)
                continue
            
            self.theta -= step*self._MSE_Gradient(X, y, y_pred)
        
        return (self.theta, loss_prev)

    def _MSE(self, y, y_pred):
        return np.square(y - y_pred).mean() * (1 / 2)
    
    def _MSE_Gradient(self, X, y, y_pred):
        N, M = X.shape
        return (X.T.dot(y - y_pred)) * (-1 / N)
        
    def predict(self, X):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return np.dot(X, self.theta) 
