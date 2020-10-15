#!/usr/bin/env python3
"""performs a valid convolution on grayscale images"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """performs a valid convolurion on grayscale image"""

    w, h, m = images.shape[2], images.shape[1], images.shape[0]
    kw, kh = kernel.shape[1], kernel.shape[0]
    ph = padding[0]
    pw = padding[1]
    images_padded = np.pad(images,
                           pad_width=((0, 0), (ph, ph), (pw, pw)),
                           mode='constant', constant_values=0)
    new_h = int(images_padded.shape[1] - kh + 1)
    new_w = int(images_padded.shape[2] - kw + 1)
    output = np.zeros((m, new_h, new_w))

    for x in range(new_w):
        for y in range(new_h):
            output[:, y, x] = \
                (kernel * images_padded[:,
                                        y: y + kh,
                                        x: x + kw]).sum(axis=(1, 2))

    return output
