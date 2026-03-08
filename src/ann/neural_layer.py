"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np


class NeuralLayer:
    """
    Fully-connected linear layer:
        Z = A_prev @ W + b

    Stores:
        W, b
        grad_W, grad_b
    """

    def __init__(self, input_dim, output_dim, weight_init="xavier", seed=42):
        np.random.seed(seed)

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Weight initialization
        if weight_init == "xavier":
            limit = np.sqrt(2.0 / (input_dim + output_dim))
            self.W = np.random.randn(input_dim, output_dim) * limit
            self.b = np.zeros((1, output_dim))

        elif weight_init == "zeros":
            self.W = np.zeros((input_dim, output_dim))
            self.b = np.zeros((1, output_dim))

        elif weight_init == "random":
            self.W = np.random.randn(input_dim, output_dim) * 0.01
            self.b = np.zeros((1, output_dim))

        else:
            raise ValueError("Unsupported weight initialization method")

        self.b = np.zeros((1, output_dim), dtype=np.float64)
       
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        
        self.A_prev = None

    def forward(self, A_prev):
        """
        Forward pass:
            Z = A_prev @ W + b

        A_prev: (N, input_dim)
        Returns:
            Z: (N, output_dim)
        """
        self.A_prev = A_prev
        Z = A_prev @ self.W + self.b
        return Z

    def backward(self, dZ, weight_decay=0.0):
        """
        Backward pass:

        Given:
            dZ = dL/dZ  (already averaged over batch)

        Computes:
            grad_W = A_prev^T @ dZ + lambda * W
            grad_b = sum(dZ)
            dA_prev = dZ @ W^T
        """
        
        self.grad_W =( self.A_prev.T @ dZ )     
        if weight_decay > 0:
            self.grad_W += weight_decay * self.W        
        self.grad_b = np.sum(dZ, axis=0) 
        dA_prev = dZ @ self.W.T
        return dA_prev