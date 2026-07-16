import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        z1 = (x @ np.array(W1).T) + b1
        a1 = np.maximum(0, z1)

        z2 = (a1 @ np.array(W2).T) + b2

        diff = z2 - y_true
        loss = np.mean(diff ** 2)
        N = len(y_true)
        dLz2 = (2 / N) * (z2 - y_true)
        dLW2 = np.outer(dLz2,  a1)
        dLb2 = dLz2
        dLa1 = dLz2.T @ W2
        t1 = np.where(z1 > 0, 1, 0)
        dLz1 = dLa1 * t1
        dLW1 = np.outer(dLz1.T, x)
        dLb1 = dLz1

        return {"loss": round(loss, 4), 
                "dW1": np.round(dLW1, 4), 
                "db1": np.round(dLb1, 4), 
                "dW2": np.round(dLW2, 4), 
                "db2": np.round(dLb2, 4) }

