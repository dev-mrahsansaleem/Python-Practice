from sklearn.datasets import load_digits
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from mlxtend.data import loadlocal_mnist

import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split


def cost_fun(y_true, y_pre):
    return np.mean((y_true-y_pre)**2)


class logistic_regration:

    # yhat= wx + b   => w -> slop and b -> shift along y-axis in 2d
    # cost function (calculate as low as low)=>    (sum_for_all_data_samples((actual-approx)**2))/number_of_samples
    # gradiant descent to calculate w and b

    def __init__(self, lr=0.0001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None

    # def __del__(self):
    #     print("meeeeeeeeeeee")

    def _sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    # training set X and training labels y
    # X=> m x n           y => m-vector

    def fit(self, X, y):
        # init variables
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        # gradient
        for testtest in range(self.n_iters):
            print(testtest)  # testing counter
            # yhat=wx+b
            y_hat = np.dot(X, self.w)+self.b
            y_pre = self._sigmoid(y_hat)

            dw = (1/n_samples)*(np.dot(X.T, (y_pre-y)))
            db = (1/n_samples)*(np.sum(y_pre-y))

            self.w -= self.lr*dw
            self.b -= self.lr*db

    def predict(self, X):
        y_hat = np.dot(X, self.w)+self.b
        y_pre = self._sigmoid(y_hat)
        y_pre_cls = [1 if i > 0.5 else 0 for i in y_pre]
        return y_pre_cls


digits = load_digits()
print(dir(digits))
print(digits.data)
