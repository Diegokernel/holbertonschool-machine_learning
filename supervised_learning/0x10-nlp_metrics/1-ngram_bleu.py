#!/usr/bin/env python3
"""
Function calculates the n-gram BLEU score for a sentence
"""

import numpy as np


def ngram_bleu(references, sentence, n):
    """
    calculates n-gram BLEU score for a sentence
    """

    refLen = []
    clipped = {}
    sentNgrams = [' '.join([str(jd) for jd in sentence[id:id + n]])
                  for id in range(len(sentence) - (n - 1))]

    candNlen = (len(sentNgrams))

    for refs in references:
        refNgrams = [' '.join([str(jd) for jd in refs[id:id + n]])
                     for id in range(len(sentence) - (n - 1))]

        refLen.append(len(refs))

        for w in refNgrams:
            if w in sentNgrams:
                if not clipped.keys() == w:
                    clipped[w] = 1

    ccount = sum(clipped.values())
    closest_refLen = min(refLen, key=lambda m: abs(m - candNlen))

    if candNlen > closest_refLen:
        bp = 1
    else:
        bp = np.exp(1 - (closest_refLen / len(sentence)))

    bleuScore = bp * np.exp(np.log(ccount / candNlen))

    return bleuScore
