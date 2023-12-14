import numpy as np
import unittest
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pnuopt.alg import (UnivariateLinearRegression,
                        MultivariateLinearRegression,
                        MultivariateLinearRegressionWithGradientDescent,
                        KNNRegressor)


class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        self.dataset = load_diabetes()
        #print(type(self.dataset.data), type(self.dataset.target))
        #print(self.dataset.data.shape, self.dataset.target.shape)
        self.X_train, self.X_test, self.y_train, self.y_test =\
            train_test_split(self.dataset.data, self.dataset.target)
        self.X_train_uni, self.X_test_uni = self.X_train[:, 2], self.X_test[:, 2]

    def test_univariate(self):
        uni_regression = UnivariateLinearRegression()
        uni_regression.fit(self.X_train_uni, self.y_train)
        y_pred = uni_regression.predict(self.X_test_uni)

        print('Coefficients: \n', uni_regression.w1)
        print('Intercept: \n', uni_regression.w0)
        print('Mean squared error: %.2f' % np.mean((y_pred - self.y_test) ** 2))
        self.assertLess(mean_squared_error(self.y_test, y_pred), 4000)

    def test_multivariate(self):
        regression = MultivariateLinearRegression()
        regression.fit(self.X_train, self.y_train)
        y_pred = regression.predict(self.X_test)

        print('Weights: \n', regression.weights)
        print('Mean squared error: %.2f' % np.mean((y_pred - self.y_test) ** 2))
        self.assertLess(mean_squared_error(self.y_test, y_pred), 30000)

    def test_GradientDescent(self):
        regression = MultivariateLinearRegressionWithGradientDescent()
        regression.fit(self.X_train, self.y_train)
        y_pred = regression.predict(self.X_test)

        print('Weights: \n', regression.weights)
        print('Bias: \n', regression.bias)
        print('Mean squared error: %.2f' % np.mean((y_pred - self.y_test) ** 2))
        self.assertLess(mean_squared_error(self.y_test, y_pred), 4000)

    def test_KNNRegressor(self):
        knn_regressor = KNNRegressor(k=5)
        knn_regressor.fit(self.X_train, self.y_train)
        y_pred = knn_regressor.predict(self.X_test)

        print("Mean squared error: %.2f" % mean_squared_error(self.y_test, y_pred))
        self.assertLess(mean_squared_error(self.y_test, y_pred), 4000)
