import numpy as np
from proj1_helpers import *

class LogisticRegression:
    def __init__(self, lambda_=0, fit_intercept=True):
        self.lambda_ = lambda_
        self.fit_intercept = fit_intercept
    
    def fit(self, X, y, step=0.08, threshold=1e-8, max_iter=1000000, batch_size=-1, initial_w=0):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        
        loss_prev = np.inf
        self.theta = np.zeros(X.shape[1])
        if (np.any(initial_w)):
            # flatten to ensure algorithm works e.g need (N,) instead of (N,1)
            if len(initial_w.shape) > 1:
                initial_w = initial_w.flatten()
            self.theta = initial_w

        it = None
        if batch_size != -1:
            it = batch_iter(y, X, batch_size, num_batches=max_iter)
        
        for _ in range(max_iter):
            y_pred = sigmoid(np.dot(X, self.theta))
            loss = self._NLL(X, y, y_pred)
            if loss_prev - loss < threshold:
                break
            loss_prev = loss
            
            if batch_size != -1:
                y_batch, X_batch = next(it)
                y_pred_batch = sigmoid(np.dot(X_batch, self.theta))
                self.theta -= step*self._NLL_Gradient(X_batch, y_batch, y_pred_batch)
                continue
            
            self.theta -= step*self._NLL_Gradient(X, y, y_pred)
        
        return (self.theta, loss_prev)
            
    def _NLL(self, X, y, y_pred):
        nll = -np.log(y_pred[y == 1]).sum() - np.log(1 - y_pred[y == 0]).sum()
        penalty = 0.5 * self.lambda_ * np.linalg.norm(self.theta, 2) ** 2
        return (penalty + nll) / X.shape[0]
    
    def _NLL_Gradient(self, X, y, y_pred):
        d_penalty = self.lambda_ * self.theta
        return -(np.dot(y - y_pred, X) + d_penalty) / X.shape[0]    
    
    def predict(self, X):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return sigmoid(np.dot(X, self.theta))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
