from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from mlxtend.data import loadlocal_mnist

import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split


def cost_fun(y_true, y_pre):
    return np.mean((y_true-y_pre)**2)


class linear_regration:

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

            dw = (1/n_samples)*(np.dot(X.T, (y_hat-y)))
            db = (1/n_samples)*(np.sum(y_hat-y))

            self.w -= self.lr*dw
            self.b -= self.lr*db

    def predict(self, X):
        return np.dot(X, self.w)+self.b


# data sets
X_test, y_test = loadlocal_mnist(
    images_path='./dataset/t10k-images-idx3-ubyte',
    labels_path='./dataset/t10k-labels-idx1-ubyte')
X_train, y_train = loadlocal_mnist(
    images_path='./dataset/train-images-idx3-ubyte',
    labels_path='./dataset/train-labels-idx1-ubyte')


# print('Dimensions of test data set: %s x %s' %
#       (X_test.shape[0], X_test.shape[1]))

# print('Dimensions of train data set: %s x %s' %
#       (X_train.shape[0], X_train.shape[1]))

# print('Digits:  0 1 2 3 4 5 6 7 8 9')
# print('labels: %s' % np.unique(y_train))
# print('Class distribution: %s' % np.bincount(y_train))
# print(y_test)

# cmap = ListedColormap(['#00ff00', '#ff00ff', '#ff0000'])
# plt.figure()
# plt.scatter(X_test[:, 0], X_test[:, 5], c=y_test,
#             cmap=cmap, edgecolors='k', s=20)
# plt.show()

# testing 100 * 1 data set

# X, y, = datasets.make_regression(
#     n_samples=100, n_features=1, noise=30, random_state=30)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=1234)

test = linear_regration()
test.fit(X_train, y_train)
pred_line = test.predict(X_test)
cost = cost_fun(y_test, pred_line)
print(cost)


test1 = linear_regration(lr=0.1)
test1.fit(X_train, y_train)
pred_line1 = test1.predict(X_test)
cost1 = cost_fun(y_test, pred_line1)
print("new", cost1)

cmap = plt.get_cmap('viridis')
fig = plt.figure()
m1 = plt.scatter(X_train, y_train, color='green', s=10)
m2 = plt.scatter(X_test, y_test, color='blue', s=10)
plt.plot(X_test, pred_line, color='black', linewidth=2)
plt.plot(X_test, pred_line1, color='red', linewidth=2)
plt.show()

# print(X)
# print(y)
# print("y=> ", y)
# print("X=> ", X[:, 0])
# print(y.shape)
# print(X.shape)
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

# fig = plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], y, color="r", marker='o', s=30)
# plt.show()
