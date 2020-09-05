#!/usr/bin/env python3
"""
RNN
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN
    """
    T, m, i = X.shape
    _, h = h_0.shape

    H = np.zeros((T + 1, m, h))
    H[0] = h_0
    h_next = H[0]
    Y = []

    for t in range(T):
        h_next, y = rnn_cell.forward(h_next, X[t])
        H[t + 1] = h_next
        Y.append(y)

    return H, np.array(Y)
