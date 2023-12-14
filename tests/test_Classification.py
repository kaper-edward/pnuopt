import unittest

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from pnuopt.alg import KNNClassifier

class TestClassification(unittest.TestCase):
    def setUp(self):
        self.dataset = load_iris()
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.dataset.data, self.dataset.target)

    def test_KNNClassifier(self):
        knn_classifier = KNNClassifier(k=5)
        knn_classifier.fit(self.X_train, self.y_train)
        y_pred = knn_classifier.predict(self.X_test)

        print("Accuracy: %.2f" % accuracy_score(self.y_test, y_pred))
