"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np


class CrossEntropyLoss:
    """
    Cross entropy loss for multi-class classification.
    This class internally applies softmax to logits.
    """

    def __init__(self):
        self.probs = None      
        self.targets = None    

    def forward(self, logits, y_true):
        """
        logits : shape (N, C)
        y_true : one-hot labels, shape (N, C)

        Returns:
            scalar loss (average over batch)
        """
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)

        exp_values = np.exp(shifted_logits)
        softmax_output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.probs = softmax_output
        self.targets = y_true

        eps = 1e-12
        log_probs = np.log(softmax_output + eps)

        batch_size = logits.shape[0]
        loss = -np.sum(y_true * log_probs) / batch_size

        return loss

    def backward(self):
        """
        Returns:
            dL/dZ  (gradient w.r.t logits)

        Since loss is averaged, gradient is divided by batch size.
        """

        batch_size = self.targets.shape[0]
        d_logits = (self.probs - self.targets) / batch_size

        return d_logits


class MSELoss:
    """
    Mean Squared Error loss and Softmax is not required for this.
    """

    def __init__(self):
        self.predictions = None
        self.targets = None

    def forward(self, predictions, y_true):
        """
        predictions : shape (N, C)
        y_true      : shape (N, C)

        Returns:
            scalar loss (average over batch)
        """

        self.predictions = predictions
        self.targets = y_true

        batch_size = predictions.shape[0]

        # squared error
        loss = np.sum((predictions - y_true) ** 2) / batch_size

        return loss

    def backward(self):
        """
        Returns:
            dL/d(predictions)
        """

        batch_size = self.targets.shape[0]
        d_pred = 2 * (self.predictions - self.targets) / batch_size

        return d_pred