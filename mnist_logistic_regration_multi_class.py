from mlxtend.data import loadlocal_mnist
import numpy as np


class LogisticRegression():

    def set_values(self, initial_params, alpha=0.01, max_iter=5000, class_of_interest=0):
        """Set the values for initial params, step size, maximum iteration, and class of interest"""
        self.params = initial_params
        self.alpha = alpha
        self.max_iter = max_iter
        self.class_of_interest = class_of_interest

    @staticmethod
    def _sigmoid(x):

        return 1.0 / (1.0 + np.exp(-x))

    def predict(self, x_bar, params):

        return self._sigmoid(np.dot(params, x_bar))

    def _compute_cost(self, input_var, output_var, params):

        cost = 0
        for x, y in zip(input_var, output_var):
            x_bar = np.array(np.insert(x, 0, 1))
            y_hat = self.predict(x_bar, params)

            y_binary = 1.0 if y == self.class_of_interest else 0.0
            cost += y_binary * np.log(y_hat) + \
                (1.0 - y_binary) * np.log(1 - y_hat)

        return cost

    def train(self, input_var, label, print_iter=5000):

        iteration = 1
        while iteration < self.max_iter:
            print(f'iteration: {iteration}')
            print(
                f'cost: {self._compute_cost(input_var, label, self.params)}')
            print('--------------------------------------------')

            for i, xy in enumerate(zip(input_var, label)):

                x_bar = np.array(np.insert(xy[0], 0, 1))
                y_hat = self.predict(x_bar, self.params)

                y_binary = 1.0 if xy[1] == self.class_of_interest else 0.0
                gradient = (y_binary - y_hat) * x_bar
                self.params += self.alpha * gradient

            iteration += 1

        return self.params

    def test(self, input_test, label_test):
        self.total_classifications = 0
        self.correct_classifications = 0

        for x, y in zip(input_test, label_test):
            self.total_classifications += 1
            x_bar = np.array(np.insert(x, 0, 1))
            y_hat = self.predict(x_bar, self.params)
            y_binary = 1.0 if y == self.class_of_interest else 0.0

            if y_hat >= 0.5 and y_binary == 1:
                # correct classification of class_of_interest
                self.correct_classifications += 1

            if y_hat < 0.5 and y_binary != 1:
                # correct classification of an other class
                self.correct_classifications += 1

        self.accuracy = self.correct_classifications / self.total_classifications

        return self.accuracy


X_test, y_test = loadlocal_mnist(
    images_path='./dataset/t10k-images-idx3-ubyte',
    labels_path='./dataset/t10k-labels-idx1-ubyte')
X_train, y_train = loadlocal_mnist(
    images_path='./dataset/train-images-idx3-ubyte',
    labels_path='./dataset/train-labels-idx1-ubyte')


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
X_train /= 255.0
X_test /= 255.0

print("train=>", X_train.shape)
print("train=>", y_train.__len__())
print("test=>", X_test.shape)
print("test=>", y_test.__len__())

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

output = list()

alpha = 1e-2
params_0 = np.zeros(len(X_test[0]+X_test[0]) + 1)
max_iter = 10

regression_model = LogisticRegression()
for x in range(10):
    regression_model.set_values(params_0, alpha, max_iter, x)

    params =\
        regression_model.train(
            X_train, y_train, 1000)

    accuracy = regression_model.test(
        X_test, y_test)
    output.append(accuracy)
    print(
        f'Accuracy of prediciting a {x} digit in test set: {accuracy}')

for x in range(10):
    print("| ", output[x], " |")
