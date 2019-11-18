# Machine Learning - Project 1, EPFL

## Group:
Rahul Rajesh, Zhuang Xinjie, Chitrangna Bhatt

## Background:

https://higgsml.lal.in2p3.fr/files/2014/04/documentation_v1.8.pdf

## Objective:

Estimate the likelihood that a given event's signature was the result of a Higgs boson.

## Files Attached:

1. `linear_models` (directory) : this folder contains all the implementation details for the models used. It is written in an object-oriented fashion for ease of use.

    - `linear_reg_gd.py` : implementation of linear regression using gradient descent
    - `linear_reg_lsq.py` : implementation of linear regression using normal equation
    - `logistic_reg.py` : implementation of logistic regression using gradient equation
    - `ridge_reg_lsq.py` : implementation of ridge regression using normal equation

2. `preprocess` (directory) : this folder contains all the implementation details for the various preprocessing teachniques used.

    - `imputer.py` : class that handles imputation for missing or undefined values
    - `scaler.py` : class that handles normalization of feature matrix

3. `implementations.py`: method implementations required for the project. Note this file relises on the `linear_models` directory

4. `proj1_helpers.py`: various helper methods for the project

5. `project1.ipynb`: jupyter notebook showcasing the various steps carried out to solve this problem

6. `run.py`: generates csv file for test set - used for submission to platform

7. `report.pdf`: final report for project


## How to reproduce predictions:

Requirements: Python3, Numpy, Matplotlib

1. Specify input path for data-files in `run.py`
2. Specify output path for prediction in `run.py`
3. In your terminal, run `python run.py`

