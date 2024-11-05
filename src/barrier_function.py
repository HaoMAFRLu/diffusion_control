"""Class for the barrier function
"""
import torch

class BarrierFunction():
    """
    """
    def __init__(self, radius=7, center=(22, 14), epsilon=1e-2):
        self.radius = radius
        self.center = center
        self.epsilon = epsilon
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def preprocess(self, x):
        return x.clamp(min=0.0, max=1.0)

    def barrier_gradient(self, x):
        x_normalized = self.preprocess(x)
        nr, channel, height, width = x_normalized.shape
        
        Y, X = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        dist_from_center = (X - self.center[0]) ** 2 + (Y - self.center[1]) ** 2
        mask = dist_from_center <= self.radius ** 2 

        gradient_V = torch.zeros_like(x_normalized).to(self.device)
        gradient_V[:, :, mask] = -1 / (x_normalized[:, :, mask] + self.epsilon) ** 2

        return gradient_V