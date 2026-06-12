import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        n = y_true.size
        pred = (1e-7) + y_pred
        first = y_true * np.log(pred)
        second = (1-y_true) * np.log(1-pred)
        res = -np.mean(first + second)
        return round(res, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        n = y_true.size
        pred = (1e-7) + y_pred
        res = -np.mean(np.sum(y_true * np.log(pred), axis=1))
        return round(res, 4)


