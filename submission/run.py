import numpy as np
import matplotlib.pyplot as plt

# import helpers
from proj1_helpers import *

# all linear models
from linear_models.linear_reg_gd import * 
from linear_models.linear_reg_lsq import *
from linear_models.logistic_reg import *
from linear_models.ridge_reg_lsq import *

# preprocessing
from preprocess.imputer import *
from preprocess.scaler import *

# Import Data
DATA_TRAIN_PATH = '../data/train.csv' # download train data and supply path here 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH, sub_sample=False)

#different encoding just for logistic regression!
log_y = np.array(y, copy=True)
log_y[log_y==-1] = 0

''' PREPROCESS '''

# Create inverse log values of features which is positive in value.
def inverse_log_cols(X, X_test):
    inv_log_cols = (0,1,2,3,4,5,7,8,9,10,12,13,16,19,21,23,26)
    X_inv_log_cols = np.log(1 / (1 + X[:, inv_log_cols]))
    X = np.hstack((X, X_inv_log_cols))
    X_test_inv_log_cols = np.log(1 / (1 + X_test[:, inv_log_cols]))
    X_test = np.hstack((X_test, X_test_inv_log_cols))
    return X, X_test

# expands X by concatenating it with powers of itself (powers used are 2, 3 ... degree)
def build_poly(X, degree):
    ''' Polynomial basis functions from input data x '''
    N, M = X.shape
    for d in range(2, degree+1):
        for i in range(M):
            col = X[:, i]
            X = np.c_[X, np.power(col, d)]
    return X

def preprocess(X, X_test, degree, strategy="most frequent"):    
    # Impute -999 with a strategy - most frequent/median/mean
    imputer = Imputer(-999, strategy)
    X = imputer.fit_transform(X)
    X_test = imputer.transform(X_test)
    
    # Log transformation
    X, X_test = inverse_log_cols(X, X_test)
    
    # extend matrix
    X = build_poly(X, degree)
    X_test = build_poly(X_test, degree)
    
    # Normalize values
    scaler = Scaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)
    
    return X, X_test

def build_k_indices(num_row, k_fold):
    """build k indices for k-fold."""
    interval = int(num_row / k_fold)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


''' CROSS - VALIDATION '''

"""
Main function that does Cross-Validation. Explanation of parameters:

model : type of model to use
X : features
y : labels
degree : max degree for expansion with polynomial basis
assign_labels: function that assigns [-1,1] or [0,1] based on value/cutoff
k_fold: how much to split the data for cross validation, defaults to 4
seed: seed for randomization
cutoff: at what point do we want to assign 1 or -1/0
strategy: strategy for imputation of undefined/missin values
"""
def cross_validate(model, X, y, degree, assign_labels, k_fold=4, seed=1, cutoff=0, strategy="most frequent"):
    np.random.seed(seed) 
    k_indices = build_k_indices(y.shape[0], k_fold)
    test_acc = []
    train_acc = []
    
    def run_cv(k):
        k_test = k_indices[k]
        
        mask = np.ones(len(k_indices), dtype=bool)
        mask[[k]] = False
        k_train = k_indices[mask].ravel()
        
        X_train, y_train = X[k_train], y[k_train]
        X_test, y_test = X[k_test], y[k_test]
                
        X_train, X_test = preprocess(X_train, X_test, degree, strategy)
        
        model.fit(X_train, y_train)
  
        y_pred = assign_labels(model.predict(X_test), cutoff=cutoff)        
        test_acc.append(compute_accuracy(y_pred, y_test))
        
        y_pred_train = assign_labels(model.predict(X_train), cutoff=cutoff)
        train_acc.append(compute_accuracy(y_pred_train, y_train))
        
    for k in range(k_fold):
        run_cv(k)
    
    return np.array(train_acc), np.array(test_acc)

# ridge model solver
def ridge_reg_solver(tX, y, tX_test):
    lambda_ = 1.00000000e-10
    degree = 15
    model = RidgeRegressionLS(lambda_=lambda_)    
    tX, tX_test = preprocess(tX, tX_test, degree)         
    model.fit(tX, y)
    y_pred = predict_labels(model.predict(tX_test))
    return y_pred

# log reg solver (final model)
def log_reg_solver(tX, y, tX_test):
    lambda_ = 0.05
    degree = 2
    model = LogisticRegression(lambda_=lambda_)
    tX, tX_test = preprocess(tX, tX_test, degree) 
    model.fit(tX, y)
    y_pred = predict_labels_01(model.predict(tX_test))
    y_pred[y_pred==0] = -1
    return y_pred

DATA_TEST_PATH = '../data/test.csv' # TODO: download train data and supply path here 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

OUTPUT_PATH = 'pred.csv'

create_csv_submission(ids_test, log_reg_solver(tX, log_y, tX_test) ,OUTPUT_PATH)
