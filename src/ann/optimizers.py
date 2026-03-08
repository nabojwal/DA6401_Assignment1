"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, NAG and RMSProp.
"""

import numpy as np

class SGD:
    """
    Stochastic Gradient Descent
    """

    def __init__(self, layers, learning_rate):
        self.layers = layers
        self.lr = learning_rate

    def step(self):
        for layer in self.layers:
            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b


class Momentum:
    """
    SGD with Momentum 
    """

    def __init__(self, layers, learning_rate, beta=0.9):
        self.layers = layers
        self.lr = learning_rate
        self.beta = beta

        self.v_W = []
        self.v_b = []

        for layer in layers:
            self.v_W.append(np.zeros_like(layer.W))
            self.v_b.append(np.zeros_like(layer.b))

    def step(self):
        # print("step was called")

        for idx in range(len(self.layers)):
            layer = self.layers[idx]

            # update velocity
            self.v_W[idx] = (
                self.beta * self.v_W[idx]
                + self.lr * layer.grad_W
            )

            self.v_b[idx] = (
                self.beta * self.v_b[idx]
                + self.lr * layer.grad_b
            )
            
            layer.W = layer.W - self.v_W[idx]
            layer.b = layer.b - self.v_b[idx]


class NAG:
    """
    Nesterov Accelerated Gradient (NAG) optimizer
    """

    def __init__(self, layers, learning_rate, beta=0.9):

        self.layers = layers
        self.lr = learning_rate
        self.beta = beta

        self.v_W = []
        self.v_b = []

        for layer in layers:
            self.v_W.append(np.zeros_like(layer.W))
            self.v_b.append(np.zeros_like(layer.b))

    def step(self):

        for idx, layer in enumerate(self.layers):

            prev_v_W = self.v_W[idx]
            prev_v_b = self.v_b[idx]

            # update velocity
            self.v_W[idx] = self.beta * self.v_W[idx] + self.lr * layer.grad_W
            self.v_b[idx] = self.beta * self.v_b[idx] + self.lr * layer.grad_b

            # Nesterov update
            layer.W -= (-self.beta * prev_v_W + (1 + self.beta) * self.v_W[idx])
            layer.b -= (-self.beta * prev_v_b + (1 + self.beta) * self.v_b[idx])


class RMSProp:
    """
    RMSProp optimizer
    """

    def __init__(self, layers, learning_rate, beta=0.9, epsilon=1e-8):

        self.layers = layers
        self.lr = learning_rate
        self.beta = beta
        self.eps = epsilon

        self.v_W = []
        self.v_b = []

        for layer in layers:
            self.v_W.append(np.zeros_like(layer.W))
            self.v_b.append(np.zeros_like(layer.b))

    def step(self):

        for idx, layer in enumerate(self.layers):

            gW = layer.grad_W
            gb = layer.grad_b

            self.v_W[idx] = self.beta * self.v_W[idx] + (1 - self.beta) * (gW ** 2)
            self.v_b[idx] = self.beta * self.v_b[idx] + (1 - self.beta) * (gb ** 2)

            layer.W -= self.lr * gW / (np.sqrt(self.v_W[idx]) + self.eps)
            layer.b -= self.lr * gb / (np.sqrt(self.v_b[idx]) + self.eps)

class Adam:
    """
    Adam optimizer
    """

    def __init__(self, layers, learning_rate,
                 beta1=0.9, beta2=0.999, epsilon=1e-8):

        self.layers = layers
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon

        self.t = 0

        self.m_W = []
        self.m_b = []
        self.v_W = []
        self.v_b = []

        for layer in layers:
            self.m_W.append(np.zeros_like(layer.W))
            self.m_b.append(np.zeros_like(layer.b))
            self.v_W.append(np.zeros_like(layer.W))
            self.v_b.append(np.zeros_like(layer.b))

    def step(self):

        self.t += 1

        for idx, layer in enumerate(self.layers):

            gW = layer.grad_W
            gb = layer.grad_b

            self.m_W[idx] = self.beta1 * self.m_W[idx] + (1 - self.beta1) * gW
            self.m_b[idx] = self.beta1 * self.m_b[idx] + (1 - self.beta1) * gb

            self.v_W[idx] = self.beta2 * self.v_W[idx] + (1 - self.beta2) * (gW ** 2)
            self.v_b[idx] = self.beta2 * self.v_b[idx] + (1 - self.beta2) * (gb ** 2)

            mW_hat = self.m_W[idx] / (1 - self.beta1 ** self.t)
            mb_hat = self.m_b[idx] / (1 - self.beta1 ** self.t)

            vW_hat = self.v_W[idx] / (1 - self.beta2 ** self.t)
            vb_hat = self.v_b[idx] / (1 - self.beta2 ** self.t)

            layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
            layer.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)

class Nadam:
    """
    Nadam optimizer 
    """

    def __init__(self, layers, learning_rate,
                 beta1=0.9, beta2=0.999, epsilon=1e-8):

        self.layers = layers
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon

        self.t = 0

        self.m_W = []
        self.m_b = []
        self.v_W = []
        self.v_b = []

        for layer in layers:
            self.m_W.append(np.zeros_like(layer.W))
            self.m_b.append(np.zeros_like(layer.b))
            self.v_W.append(np.zeros_like(layer.W))
            self.v_b.append(np.zeros_like(layer.b))

    def step(self):

        self.t += 1

        for idx, layer in enumerate(self.layers):

            gW = layer.grad_W
            gb = layer.grad_b

            self.m_W[idx] = self.beta1 * self.m_W[idx] + (1 - self.beta1) * gW
            self.m_b[idx] = self.beta1 * self.m_b[idx] + (1 - self.beta1) * gb

            self.v_W[idx] = self.beta2 * self.v_W[idx] + (1 - self.beta2) * (gW ** 2)
            self.v_b[idx] = self.beta2 * self.v_b[idx] + (1 - self.beta2) * (gb ** 2)

            mW_hat = self.m_W[idx] / (1 - self.beta1 ** self.t)
            mb_hat = self.m_b[idx] / (1 - self.beta1 ** self.t)

            vW_hat = self.v_W[idx] / (1 - self.beta2 ** self.t)
            vb_hat = self.v_b[idx] / (1 - self.beta2 ** self.t)

            # Nesterov modification
            mW_nesterov = self.beta1 * mW_hat + (1 - self.beta1) * gW
            mb_nesterov = self.beta1 * mb_hat + (1 - self.beta1) * gb

            layer.W -= self.lr * mW_nesterov / (np.sqrt(vW_hat) + self.eps)
            layer.b -= self.lr * mb_nesterov / (np.sqrt(vb_hat) + self.eps)




            