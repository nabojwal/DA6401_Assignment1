"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""
import numpy as np


class ReLU:
    """
    ReLU activation: f(z) = max(0, z)
    """

    def __init__(self):
        self.Z = None  

    def forward(self, Z):
        """
        Z: (N, D)
        """
        self.Z = Z
        return np.maximum(0, Z)

    def backward(self, dA):
        """
        dA: gradient coming from next layer
        """
        dZ = dA.copy()
        dZ[self.Z <= 0] = 0
        return dZ


class Sigmoid:
    """
    Sigmoid activation
    """

    def __init__(self):
        self.A = None  # store activation output

    def forward(self, Z):
        """
        Z: (N, D)
        """
        self.A = 1 / (1 + np.exp(-Z))
        return self.A

    def backward(self, dA):
        """
        dA: gradient from next layer
        derivative of sigmoid = A * (1 - A)
        """
        return dA * self.A * (1 - self.A)


class Tanh:
    """
    Tanh activation
    """

    def __init__(self):
        self.A = None 

    def forward(self, Z):
        """
        Z: (N, D)
        """
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dA):
        """
        derivative of tanh = 1 - A^2
        """
        return dA * (1 - self.A ** 2)