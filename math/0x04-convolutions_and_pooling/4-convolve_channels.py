#!/usr/bin/env python3
"""performs a valid convolution on grayscale images"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """performs a valid convolurion on grayscale image"""

    c, w, = images.shape[3], images.shape[2]
    h, m = images.shape[1], images.shape[0]
    kw, kh = kernel.shape[1], kernel.shape[0]
    sw, sh = stride[1], stride[0]

    pw, ph = 0, 0

    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1

    if isinstance(padding, tuple):
        ph = padding[0]
        pw = padding[1]

    images_padded = np.pad(images,
                           pad_width=((0, 0),
                                      (ph, ph),
                                      (pw, pw),
                                      (0, 0)),
                           mode='constant', constant_values=0)

    new_h = int(((images_padded.shape[1] - kh) / sh) + 1)
    new_w = int(((images_padded.shape[2] - kw) / sw) + 1)
    output = np.zeros((m, new_h, new_w))
    for x in range(new_w):
        for y in range(new_h):
            output[:, y, x] = \
                (kernel * images_padded[:,
                                        y * sh: y * sh + kh,
                                        x * sw: x * sw + kw,
                                        :]).sum(axis=(1, 2, 3))

    return output
