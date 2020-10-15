#!/usr/bin/env python3
""" Script to backward propagate over a pooling layer in a NN"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Function to backward propagate over a convolutional layer in a NN
    Args:
    Returns: the partial derivatives with respect to the previous layer
             (dA_prev), the kernels (dW), and the biases (db), respectively
    """
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (m, h_new, w_new, c_new) = dZ.shape
    (kh, kw, c_prev, c_new) = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil((((h_prev - 1) * sh + kh - h_prev) / 2)))
        pw = int(np.ceil((((w_prev - 1) * sw + kw - w_prev) / 2)))
    if padding == 'valid':
        pw = 0
        ph = 0

    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    A_prev_pad = np.pad(A_prev, pad_width=((0, 0), (ph, ph), (pw, pw),
                                           (0, 0)), mode='constant')
    dA_prev_pad = np.pad(dA_prev, pad_width=((0, 0), (ph, ph), (pw, pw),
                                             (0, 0)), mode='constant')

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    v_start = h * sh
                    v_end = v_start + kh
                    h_start = w * sw
                    h_end = h_start + kw
                    a_slice = a_prev_pad[v_start:v_end, h_start:h_end]

                    # update gradients for the window filter param
                    da_prev_pad[v_start:v_end,
                                h_start:h_end] += \
                        W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]

        if padding == 'same':
            # set the ith training example dA_prev to unppaded da_prev_pad
            dA_prev[i, :, :, :] += da_prev_pad[ph:-ph, pw:-pw, :]
        if padding == 'valid':
            dA_prev[i, :, :, :] += da_prev_pad

    return dA_prev, dW, db
