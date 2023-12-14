import numpy as np


class KNNClassifier:
  def __init__(self, k=3):
    self.k = k
    self.X_train = None
    self.y_train = None

  def fit(self, X, y):
    self.X_train = X
    self.y_train = y

  def predict(self, X):
    from collections import Counter
    predictions = []
    for x in X:
      distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
      k_indices = np.argsort(distances)[:self.k]
      k_nearest_labels = self.y_train[k_indices]
      most_common = Counter(k_nearest_labels).most_common(1)
      predictions.append(most_common[0][0])

    return np.array(predictions)
