from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np


class LR:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_sample, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for i in range(self.n_iters):
            print(i)

            linear_model = np.dot(X, self.w)+self.b
            y_pridicted = self._sigmoid(linear_model)

            dw = (1/n_sample)*(np.dot(X.T, (y_pridicted-y)))
            db = (1/n_sample)*(np.sum(y_pridicted-y))

            self.w -= self.lr*dw
            self.b -= self.lr*db

    def predict(self, X):
        linear_model = np.dot(X, self.w)+self.b
        y_pridicted = self._sigmoid(linear_model)

        y_pridicted_cls = [1 if i > 0.5 else 0 for i in y_pridicted]

        return y_pridicted_cls

    def _sigmoid(self, x):
        return (1)/(1+np.exp(-x))


# bc = datasets.load_breast_cancer()
# X, y = bc.data, bc.target

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=1234)


X_test, y_test = loadlocal_mnist(
    images_path='./dataset/t10k-images-idx3-ubyte',
    labels_path='./dataset/t10k-labels-idx1-ubyte')
X_train, y_train = loadlocal_mnist(
    images_path='./dataset/train-images-idx3-ubyte',
    labels_path='./dataset/train-labels-idx1-ubyte')

y_train = [1 if i == 6 else 0 for i in y_train]
y_test = [1 if i == 6 else 0 for i in y_test]
# print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)
reg = LR(lr=0.0001, n_iters=1000)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

acc = np.sum(y_test == predictions)/len(y_test)
print('{:20.15f}'.format(acc))
