import numpy as np
from numpy.linalg import inv


class UnivariateLinearRegression:
  def __init__(self):
    self.w0 = 0
    self.w1 = 0

  def fit(self, X, y):
    N = len(X)
    sum_x = np.sum(X)
    sum_y = np.sum(y)
    sum_xy = np.sum(X * y)
    sum_x_squared = np.sum(X**2)

    # least squares
    self.w1 = (N * sum_xy - sum_x * sum_y) / (N * sum_x_squared - sum_x**2)
    self.w0 = (sum_y - self.w1 * sum_x) / N

  def predict(self, X):
    return self.w0 + self.w1 * X


class MultivariateLinearRegression:
  def __init__(self):
    self.weights = None

  def fit(self, X, y):
    self.weights = inv(X.T.dot(X)).dot(X.T).dot(y)

  def predict(self, X):
    return X.dot(self.weights)


class MultivariateLinearRegressionWithGradientDescent:
  def __init__(self, learning_rate=0.5, iterations=1000):
    self.learning_rate = learning_rate
    self.iterations = iterations
    self.weights = None
    self.bias = None

  def fit(self, X, y):
    m, n = X.shape
    self.weights = np.zeros(n)
    self.bias = 0

    for _ in range(self.iterations):
      y_pred = self.predict(X)

      dw = (2/m) * np.dot(X.T, (y_pred - y) )
      db = (2/m) * np.sum(y_pred - y)

      self.weights -= self.learning_rate * dw
      self.bias -= self.learning_rate * db

  def predict(self, X):
    return np.dot(X, self.weights) + self.bias

