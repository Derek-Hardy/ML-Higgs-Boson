from collections import Counter
import numpy as np

class Imputer:
    def __init__(self, missing_values, strategy):
        self._mv = missing_values
        
        err_msg = "strategy must be 'mean', 'median' or 'most frequent', but got: {}".format(strategy)
        assert strategy in ["mean", "median", "most frequent"], err_msg
        
        self._strategy = strategy
        
    def _mode(self, X):
        N, M = X.shape
        for i in range(M):
            col = list(filter(lambda x : x != self._mv, X[:, i]))
            mode = [self._mv]
            
            if len(col) > 0:
                counter = Counter(col)
                max_count = max(counter.values())
                mode = [k for k,v in counter.items() if v == max_count]
            
            self.cache.append(mode[0])
        
    def _mean(self, X):
        N, M = X.shape
        for i in range(M):
            miss_index = X[:, i] == -999
            mu = np.mean(X[~miss_index, i])
            self.cache.append(mu)
        
    def _median(self, X):
        N, M = X.shape
        for i in range(M):
            miss_index = X[:, i] == -999
            median = np.median(X[~miss_index, i])
            self.cache.append(median)  
            
    def fit(self, X):
        self.cache = []
        if self._strategy == "most frequent": self._mode(X)
        elif self._strategy == "mean": self._mean(X)
        elif self._strategy == "median": self._median(X)
        
    def transform(self, X):
        N, M = X.shape
        nX = np.array(X, copy=True)
        for i in range(M):
            c = nX[:, i]
            c[c == self._mv] = self.cache[i]
        
        del_c = []
        for i in range(M):
            c = nX[:, i]
            if c[c == self._mv].size > 0:
                del_c.append(i)
                
        nX = np.delete(nX, del_c, 1)
        return nX
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
