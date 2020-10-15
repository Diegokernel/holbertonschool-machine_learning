#!/usr/bin/env python3
"""
Pooling Back Prop
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network
    :param dA: is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing
    """
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    for img in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for cn in range(c_new):
                    tmp_dA = dA[img, i, j, cn]
                    tmp_A_p = A_prev[img, i * sh:i *
                                     sh + kh, j * sw:j * sw + kw, cn]
                    if mode is 'max':
                        aux = (tmp_A_p == np.max(tmp_A_p))
                    else:
                        aux = np.ones((kh, kw))
                        aux /= (kh * kw)
                    dA_prev[img, i * sh:i * sh + kh, j *
                            sw:j * sw + kw, cn] += aux * tmp_dA

    return dA_prev

