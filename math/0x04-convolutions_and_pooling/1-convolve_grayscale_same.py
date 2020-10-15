#!/usr/bin/env python3
"""performs a valid convolution on grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    performs a same convolution on grayscale images:
    :return: numpy.ndarray containing the convolved images
    """
    w, h, m = images.shape[2], images.shape[1], images.shape[0]
    kw, kh = kernel.shape[1], kernel.shape[0]
    ph = max(int((kh - 1) / 2), int(kh / 2))
    pw = max(int((kw - 1) / 2), int(kw / 2))

    images_padded = np.pad(images,
                           pad_width=((0, 0), (ph, ph), (pw, pw)),
                           mode='constant', constant_values=0)

    output = np.zeros((m, h, w))

    for y in range(h):
        for x in range(w):
            output[:, y, x] =\
                (kernel * images_padded[:,
                                        y: y + kh,
                                        x: x + kw]).sum(axis=(1, 2))

    return output
