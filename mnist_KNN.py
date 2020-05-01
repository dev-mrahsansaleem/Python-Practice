from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mlxtend.data import loadlocal_mnist
import numpy as np


def distance_fun(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))


class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, x, y):
        self.X_train = x
        self.y_train = y

    def predict(self, X):
        self.count = 1
        predicted_l = [self._predict(x) for x in X]
        return np.array(predicted_l)

    def _predict(self, x):
        # distance compute
        distance = [distance_fun(x, x_train) for x_train in self.X_train]
        # get k nearest samples
        k_index = np.argsort(distance)[:self.k]
        print(self.count)
        self.count = self.count+1
        # print(k_index)

        k_labeles = [self.y_train[i] for i in k_index]
        # print(k_labeles)
        most_common = Counter(k_labeles).most_common(1)
        # most_common => [(value,number_of_times),(value,number_of_times)]

        return most_common[0][0]


# data set
X_test, y_test = loadlocal_mnist(
    images_path='./dataset/t10k-images-idx3-ubyte',
    labels_path='./dataset/t10k-labels-idx1-ubyte')
X_train, y_train = loadlocal_mnist(
    images_path='./dataset/train-images-idx3-ubyte',
    labels_path='./dataset/train-labels-idx1-ubyte')

print('Dimensions of test data set: %s x %s' %
      (X_test.shape[0], X_test.shape[1]))

print('Dimensions of train data set: %s x %s' %
      (X_train.shape[0], X_train.shape[1]))

# print('Digits:  0 1 2 3 4 5 6 7 8 9')
# print('labels: %s' % np.unique(y_train))
# print('Class distribution: %s' % np.bincount(y_train))

# print(y_test)

# cmap = ListedColormap(['#00ff00', '#ff00ff', '#ff0000'])

# plt.figure()
# plt.scatter(X_test[:, 0], X_test[:, 5], c=y_test,
#             cmap=cmap, edgecolors='k', s=20)
# plt.show()

test = KNN(6)
test.fit(X_train, y_train)
pre1 = test.predict(X_test)


acc = np.sum(pre1 == y_test)/len(y_test)
print(acc)
