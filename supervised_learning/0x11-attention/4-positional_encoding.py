#!/usr/bin/env python3
"""
function calculates the positional encoding got a transformer
"""
import numpy as np
import tensorflow as tf


def positional_encoding(max_seq_len, dm):
    """
    calculates positional encoding for a transformer
    """
    posEncoding = np.zeros([max_seq_len, dm])

    for i in range(dm):
        for pos in range(max_seq_len):
            posEncoding[pos, i] = pos / np.power(10000, (2 * (i // 2)) / (dm))

    posEncoding[:, 0::2] = np.sin(posEncoding[:, 0::2])
    posEncoding[:, 1::2] = np.cos(posEncoding[:, 1::2])

    return posEncoding
