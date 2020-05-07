from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np


from sklearn.linear_model import LogisticRegression


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

        return np.array(y_pridicted_cls)

    def _sigmoid(self, x):
        return (1)/(1+np.exp(-x))


def accuracy(predicted_labels, actual_labels):
    diff = predicted_labels - actual_labels
    return 1.0 - (float(np.count_nonzero(diff)) / len(diff))


def confiusionMat(predictions, actualLabels):
    T_P = F_P = F_N = T_N = 0
    for p, t in zip(pree, y_test69):
        if(p == 0 and t == 0):
            #  f f case
            T_N += 1
        if(p == 1 and t == 0):
            #  t f case
            F_P += 1
        if(p == 0 and t == 1):
            #  f t case
            F_N += 1
        if(p == 1 and t == 1):
            #  t t case
            T_P += 1
    return (T_P, F_P, F_N, T_N)

# bc = datasets.load_breast_cancer()
# X, y = bc.data, bc.target

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=1234)


# reg = LR(lr=0.0001, n_iters=1000)
# reg.fit(X_train, y_train)
# pree = reg.predict(X_test)

# acc = np.sum(y_test == pree)/len(y_test)
# print('{:20.15f}'.format(acc))
# print('{:20.15f}'.format(accuracy(pree, y_test)))


X_test, y_test = loadlocal_mnist(
    images_path='./dataset/t10k-images-idx3-ubyte',
    labels_path='./dataset/t10k-labels-idx1-ubyte')
X_train, y_train = loadlocal_mnist(
    images_path='./dataset/train-images-idx3-ubyte',
    labels_path='./dataset/train-labels-idx1-ubyte')


X_train69 = X_train[np.logical_or(y_train == 6, y_train == 9)]
y_train69 = y_train[np.logical_or(y_train == 6, y_train == 9)]

X_test69 = X_test[np.logical_or(y_test == 6, y_test == 9)]
y_test69 = y_test[np.logical_or(y_test == 6, y_test == 9)]

y_train69 = [1 if i == 6 else 0 for i in y_train69]
y_test69 = [1 if i == 6 else 0 for i in y_test69]

X_train69 = X_train69.astype('float32')
X_test69 = X_test69.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
X_train69 /= 255
X_test69 /= 255
print(X_train69.shape)
print(y_train69.__len__())
print(X_test69.shape)
print(y_test69.__len__())

X_train69 = np.array(X_train69)
y_train69 = np.array(y_train69)
X_test69 = np.array(X_test69)
y_test69 = np.array(y_test69)


maxt = 30
print(X_train69[:maxt])
print(y_train69[:maxt])
print(X_test69[:maxt])
print(y_test69[:maxt])


reg = LR(lr=0.0001, n_iters=1000)
reg.fit(X_train69, y_train69)
pree = reg.predict(X_test69[0])

T_P, F_P, F_N, T_N = confiusionMat(pree, y_test69)
print("True Positive: ", T_P, "False Positive: ", F_P,
      "False Negitive", F_N, "True Negitive", T_N)
# acc = np.sum(y_test69 == pree)/len(y_test69)
# print('{:20.15f}'.format(acc))
print("Confusion Matrix accuracy:", (T_P+T_N)/(T_P+F_P + F_N + T_N))
print("Precssion: ", (T_P)/(T_P + F_P))
print("Recall:", (T_P)/(T_P+F_N))
print("Mean farmula accuracy: ", accuracy(pree, y_test69))
print(pree.shape)
print(pree)
# print("++++++++++++++++++++++++++++++++++++++++++++++++")
# logmodel = LogisticRegression()
# logmodel.fit(X_train69, y_train69)

# predictions = logmodel.predict(X_test69)
# accuu = np.sum(y_test69 == predictions)/len(y_test69)
# print('{:20.15f}'.format(accuu))
