import numpy as np

class Scaler:
    def fit(self, X):
        self.mu = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        
    def transform(self, X):
        return (X - self.mu) / self.std
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
