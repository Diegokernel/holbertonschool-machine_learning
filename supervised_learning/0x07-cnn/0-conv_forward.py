#!/usr/bin/env python3
""" a function  that performs forward propagation over a pooling layer """
import numpy as np


def conv_forward(A_prev, W, b, activation,
                 padding="same", stride=(1, 1)):
    """ Function Convolution Forward"""

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    ph = 0
    pw = 0

    if padding == 'same':
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2)
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2)

    if type(padding) == tuple:
        ph, pw = padding

    padded_img = np.pad(A_prev,
                        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        'constant')

    ch = int(((h_prev + 2 * ph - kh) / sh) + 1)
    cw = int(((w_prev + 2 * pw - kw) / sw) + 1)

    Aprev_conv_W = np.zeros((m, ch, cw, c_new))

    for j in range(ch):
        for k in range(cw):
            for n in range(c_new):
                images_slide = padded_img[:,
                                          j * sh:j * sh + kh,
                                          k * sw:k * sw + kw]
                kernel = W[:, :, :, n]
                elem_mul = np.multiply(images_slide, kernel)
                Aprev_conv_W[:, j, k, n] = elem_mul.sum(axis=(1, 2, 3))

    Z = Aprev_conv_W + b

    return activation(Z)
