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

    def __init__(self, lr=0.01, n_iters=1000):
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
        for c in range(self.n_iters):
            print(c)  # testing counter
            # yhat=wx+b
            y_hat = np.dot(X, self.w)+self.b
            y_pre = self._sigmoid(y_hat)

            dw = (1/n_samples)*(np.dot(X.T, (y_pre-y)))
            db = (1/n_samples)*(np.sum(y_pre-y))
            # print(self.w, " : ", self.b)
            self.w -= self.lr*dw
            self.b -= self.lr*db

    def predict(self, X):
        # print("from predict: ", self.w, " : ", self.b)
        y_hat = np.dot(X, self.w)+self.b
        y_pre = self._sigmoid(y_hat)
        y_pre_cls = [1 if i > 0.5 else 0 for i in y_pre]
        return y_pre_cls


# data sets
X_test, y_test = loadlocal_mnist(
    images_path='./dataset/t10k-images-idx3-ubyte',
    labels_path='./dataset/t10k-labels-idx1-ubyte')
X_train, y_train = loadlocal_mnist(
    images_path='./dataset/train-images-idx3-ubyte',
    labels_path='./dataset/train-labels-idx1-ubyte')
# X_test69 = []
# y_test69 = []
# X_train69 = []
# y_train69 = []

# print(y_test == 6)
# print(y_test == 9)

# print(y_train[y_train == 6])
# print(y_train[y_train == 9])

# print(X_train[y_train == 6])
# print(X_train[y_train == 9])

# y_6, y_9 = y_train[y_train == 6], y_train[y_train == 9]
# X_6, X_9 = X_train[y_train == 6], X_train[y_train == 9]
# print((X_6.shape))
# print((X_9.shape))


# print("++++++++++++++++++++++++++++++++++++++++++")

# print("size of :", len(X_train))
# print("++++++++++++++++++++++++++++++++++++++++++")

# print("size of :", len(y_train))
# print("++++++++++++++++++++++++++++++++++++++++++")

# for i in range(len(y_train)):
#     if y_train[i] == 6 or y_train[i] == 9:
#         # print(y_train[i])
#         X_train69.append(X_train[i])
#         y_train69.append(y_train[i])

# print(X_train69)
# print("++++++++++++++++++++++++++++++++++++++++++")

# print(y_train69)
# print("++++++++++++++++++++++++++++++++++++++++++")

# print('Dimensions of test data set: %s x %s' %
#       (X_test.shape[0], X_test.shape[1]))
# # print(y_train.shape)
# print("++++++++++++++++++++++++++++++++++++++++++")

# print('Dimensions of train data set: %s x %s' %
#       (X_train.shape[0], X_train.shape[1]))
# print("++++++++++++++++++++++++++++++++++++++++++")


# print(y_test)

# cmap = ListedColormap(['#00ff00', '#ff00ff', '#ff0000'])
# plt.figure()
# plt.scatter(X_test[:, 0], X_test[:, 5], c=y_test,
#             cmap=cmap, edgecolors='k', s=20)
# plt.show()

# testing 100 * 1 data set

# ds = datasets.load_breast_cancer()
# X, y = ds.data, ds.target

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=1234)


# training data set of 6's and 9's only
X_train69 = np.concatenate([X_train[y_train == 6], X_train[y_train == 9]])
y_train69 = np.concatenate([y_train[y_train == 6], y_train[y_train == 9]])
y_train69 = [1 if i == 6 else 0 for i in y_train69]
y_test69 = [1 if i == 6 else 0 for i in y_test]
# print("xtrain dataset contains 6 and 9 only", X_train69)
# print("ytrain dataset contains 6 and 9 only", y_train69)

# print("xtrain dataset dimentions", X_train69.shape)
# print("ytrain dataset dimentions", y_train69.shape)
sliceing = 30

# apply logistic regration
L_R = logistic_regration()
L_R.fit(X_train69, y_train69)
pre = L_R.predict(X_test)

print("train")
print(y_train69)
print("predict")
print(pre)
print("test")
print(y_test69)
accuracy = np.mean(pre == y_test69)
print(accuracy)

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

# print(type(X_train))
# print(X_train69.shape[0])
# print(y_train69.shape)

# fig = plt.figure()
# plt.scatter(X_train69, y_train69, color="r", marker='o', s=30)
# plt.show()
