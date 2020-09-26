#!/usr/bin/env python3
"""
Creates the function positional encoding
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer
    """
    poem = np.zeros((max_seq_len, dm))

    for position in range(max_seq_len):
        for i in range(0, dm, 2):
            div = np.exp(i * -np.log(10000.0) / dm)
            poem[position, i] = (
                np.sin(position * div))
            poem[position, i + 1] = (
                np.cos(position * div))

    return poem
