#!/usr/bin/env python3
"""forward propagation to a deep RNN"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Forward prop in a deep RNN
    Returns: H, Y
             H: numpy.ndarray containing all of the hidden states
             Y: numpy.ndarray containing all of the outputs
    """

    Y = []
    t, m, i = X.shape
    _, _, h = h_0.shape
    time_step = range(t)
    layers = len(rnn_cells)
    H = np.zeros((t+1, layers, m, h))
    H[0, :, :, :] = h_0

    for ts in time_step:
        for ly in range(layers):
            if ly == 0:
                h_next, y_pred = rnn_cells[ly].forward(H[ts, ly], X[ts])
            else:
                h_next, y_pred = rnn_cells[ly].forward(H[ts, ly], h_next)
            H[ts+1, ly, :, :] = h_next
        Y.append(y_pred)

    Y = np.array(Y)
    return H, Y
