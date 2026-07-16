import torch
import torch.nn as nn
import math
from typing import List
import numpy as np


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = (2 / (fan_in+fan_out)) ** .5
        res = np.round((torch.randn(fan_out,fan_in) * std), 4)
        return res.tolist()
 

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = (2 / fan_in) ** .5
        res = np.round(torch.randn(fan_out,fan_in) * std, 4)
        return res.tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)

        weights = []
        current_in_dim = input_dim
        if init_type == "xavier":
            std = (2 / (current_in_dim+hidden_dim)) ** .5
        elif init_type == "kaiming":
            std = (2 / current_in_dim) ** .5
        else:
            std = 1
        for i in range(num_layers):
            w = torch.randn(hidden_dim,current_in_dim) * std
            current_in_dim = hidden_dim
            weights.append(w)
    
        x = torch.randn(1, input_dim)
        
        stds = []
        current_in_dim = input_dim
        for i in range(num_layers):
            x = torch.matmul(x,weights[i].T)
            x = torch.relu(x)
            stds.append(round(float(torch.std(x)), 2))

        return stds
