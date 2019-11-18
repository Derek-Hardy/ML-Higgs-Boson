# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
import matplotlib.pyplot as plt

""" I/O """

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

""" predictions for [-1, 1] and [0, 1] """

def predict_labels(y_pred, cutoff=0):
    y_pred[np.where(y_pred <= cutoff)] = -1
    y_pred[np.where(y_pred > cutoff)] = 1
    return y_pred

def predict_labels_01(y_pred, cutoff=0.5):
    y_pred[np.where(y_pred <= cutoff)] = 0
    y_pred[np.where(y_pred > cutoff)] = 1
    return y_pred

""" accuracy """

def compute_accuracy(pred, actual):
    return np.sum(pred == actual) / len(actual)

""" clean a column that has all -999 """   

def remove_single_col(X, X_train):
    N, M = X.shape
    to_del = []
    for i in range(M):
        col = X[:, i]
        if ((col - col.mean()).mean()) == 0: 
            to_del.append(i)
    return np.delete(X, to_del, 1), np.delete(X_train, to_del, 1)

""" for stochastic gradient descent """ 

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


""" for figure plotting """ 

def plot_bar_graph(headers, x1_data, x2_data, x1_label, x2_label, y_label, title):
    x = np.arange(len(headers))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()

    ax.bar(x - width/2, x1_data, width, label=x1_label)
    ax.bar(x + width/2, x2_data, width, label=x2_label)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(headers)
    ax.legend()

    fig.tight_layout()

    plt.show()
    
def plot_line_graph(title, y, t1, t2, t1_label, t2_label, y_label, x_label):
    plt.plot(y, t1, 'b-', label=t1_label)
    plt.plot(y, t2, 'r-', label=t2_label)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend(loc='upper left')
    plt.title(title)
    plt.show()
