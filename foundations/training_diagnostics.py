import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        res = []
        with torch.no_grad():
            for layer in model:
                x = layer(x)

                if isinstance(layer, nn.Linear):
                    mean = round(float(torch.mean(x).item()), 4)
                    std = round(float(torch.std(x).item()), 4)
                    dead_fraction = round((x <= 0).all(dim=0).float().mean().item(), 4)
                    curr_stats = {
                            "mean":mean, 
                            "std":std, 
                            "dead_fraction": dead_fraction
                        }

                    res.append(curr_stats)
        return res

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        
        model.zero_grad()

        for layer in model:
            x = layer(x)

        loss = nn.functional.mse_loss(x,y)
        loss.backward()
        res = []
        for layer in model:
            if isinstance(layer, nn.Linear) and layer.weight.grad is not None:
                mean = round(float(torch.mean(layer.weight.grad)), 4)
                std = round(float(torch.std(layer.weight.grad)), 4)
                norm = round(float(torch.norm(layer.weight.grad)), 4)
                curr_stats = {
                    "mean": mean,
                    "std": std,
                    "norm": norm
                }
                res.append(curr_stats)
        return res


    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)

        
        if any([act_stat["dead_fraction"] > .5 for act_stat in activation_stats]):
            return 'dead_neurons'
        if any([grad_stat["norm"] > 1000 for grad_stat in gradient_stats]):
            return 'exploding_gradients'
        if gradient_stats[-1]["norm"] < 1e-5:
            return 'vanishing_gradients'
        
        stds = [act_stat["std"] for act_stat in activation_stats]
        if min(stds) < 0.1:
            return 'vanishing_gradients'
        elif max(stds) > 10.0:
            return 'exploding_gradients'

        return "healthy"