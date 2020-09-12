#!/usr/bin/env python3
"""
This module has biderectional cell
"""
import numpy as np


class BidirectionalCell:
    """
    This class represents a bidirectional cell of an RNN
    """

    def __init__(self, i, h, o):
        """
        constructor
        """
        self.Whf = np.random.randn(i + h, h)
        self.bhf = np.zeros((1, h))
        self.Whb = np.random.randn(i + h, h)
        self.bhb = np.zeros((1, h))
        self.Wy = np.random.randn(2 * h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        The output of the cell should use a softmax activation function
        Returns: h_next, y
        """
        h1 = np.tanh(np.dot(np.concatenate([h_prev, x_t],
                                           axis=1),
                            self.Whf) + self.bhf)
        h2 = np.tanh(np.dot(np.concatenate([h_prev, x_t],
                                           axis=1),
                            self.Whb) + self.bhb)
        return h1

    def backward(self, h_next, x_t):
        """
        This method calculates the hidden state in the backward
        direction for one time step
        """
        h2 = np.tanh(np.dot(np.concatenate([h_next, x_t],
                                           axis=1),
                            self.Whb) + self.bhb)
        return h2
