import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        maximum = max(z)
        sum_exp_logits = sum(np.exp(z - maximum))

        z = np.round((np.exp(z-maximum) / sum_exp_logits), 4)
        return z