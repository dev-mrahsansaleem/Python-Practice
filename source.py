from mlxtend.data import loadlocal_mnist
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

        # y_pridicted_cls = [1 if i > 0.5 else 0 for i in y_pridicted]

        return 1 if y_pridicted > 0.5 else 0

    def _sigmoid(self, x):
        return (1)/(1+np.exp(-x))


X_test, y_test = loadlocal_mnist(
    images_path='./dataset/t10k-images-idx3-ubyte',
    labels_path='./dataset/t10k-labels-idx1-ubyte')
X_train, y_train = loadlocal_mnist(
    images_path='./dataset/train-images-idx3-ubyte',
    labels_path='./dataset/train-labels-idx1-ubyte')


# X_train69 = X_train[np.logical_or(y_train == 6, y_train == 9)]
# y_train69 = y_train[np.logical_or(y_train == 6, y_train == 9)]

# X_test69 = X_test[np.logical_or(y_test == 6, y_test == 9)]
# y_test69 = y_test[np.logical_or(y_test == 6, y_test == 9)]

# y_train69 = [1 if i == 6 else 0 for i in y_train69]
# y_test69 = [1 if i == 6 else 0 for i in y_test69]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
X_train /= 255
X_test /= 255

print("train=>", X_train.shape)
print("train=>", y_train.__len__())
print("test=>", X_test.shape)
print("test=>", y_test.__len__())

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)


# maxt = 30
# print(X_train[:maxt])
# print(y_train[:maxt])
# print(X_test[:maxt])
# print(y_test[:maxt])
output = np.zeros((y_test.__len__()+20, 10))  # 10020 x 10

reg = LR(lr=0.0001, n_iters=1000)
for x in range(10):
    y_train_s_num = [1 if i == x else 0 for i in y_train]
    y_test_s_num = [1 if i == x else 0 for i in y_test]
    records = 30
    print("itration no:", x)
    print("updated:", np.array(y_train_s_num[:records]))
    print("old____:", y_train[:records])
    print("updated:", np.array(y_test_s_num[:records]))
    print("old____:", y_test[:records])
    reg.fit(X_train, y_train_s_num)
    i = 0
    for x_t in X_test:
        # itrate over each record
        output[i][x] = reg.predict(x_t)
        i += 1
        # print("predicted for train iter:", x, "   ", output[i][x])
        # print("orignal for train iter:", x, "     ", y_test[i])

for y in range(10020):
    for x in range(10):
        print("| ", output[y][x], " |")
