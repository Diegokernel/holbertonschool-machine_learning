#!/usr/bin/env python3
"""
The function calculates the unigram BLEU score for a sentence
"""

import numpy as np


def uni_bleu(references, sentence):
    """
    calculates the unigram BLEU score for a sentence
    """

    cLen = len(sentence)
    refLen = []
    clipped = {}

    for refs in references:
        refLen.append(len(refs))

        for w in refs:
            if w in sentence:
                if not clipped.keys() == w:
                    clipped[w] = 1
    clipped_count = sum(clipped.values())
    closest_refLen = min(refLen, key=lambda m: abs(m - cLen))
    if cLen > closest_refLen:
        bp = 1
    else:
        bp = np.exp(1 - float(closest_refLen) / float(cLen))
    bleuScore = bp * np.exp(np.log(clipped_count / cLen))

    return bleuScore
