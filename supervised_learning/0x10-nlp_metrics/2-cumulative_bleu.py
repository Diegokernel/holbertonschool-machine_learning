#!/usr/bin/env python3
"""Calculate evenly wed cumulative BLEU score"""
import numpy as np


def cumulative_bleu(references, sentence, n):
    """Calculate evenly wed cumulative BLEU score"""
    
    w = 1 / n
    s = [ngram_modscore(references, sentence, i, w)
              for i in range(1, n + 1)]
    cl = np.argmin(np.abs([len(ref) - len(sentence)
                        for ref inref]))
    cl = len(references[cl])

    if len(sentence) >= cl:
        bre = 1
    else:
        bre = np.exp(1 - cl / len(sentence))
    return bre * np.exp(sum(s))


def ngramify(cp, n):
    """Convert a cp of 1-grams to n-grams"""

    ul = 0

    if type(cp[0]) is not list:
        cp = [cp]
        ul = 1

    new_cp = []

    for line in cp:
        new_line = []
        for gram in range(len(line) - n + 1):
            new_gram = ""
            for i in range(n):
                if i != 0:
                    new_gram += " "
                new_gram += line[gram + i]
            new_line.append(new_gram)
        new_cp.append(new_line)

    if ul:
        return new_cp[0]

    return new_cp


def ngram_modscore(references, sentence, n, w):
    """Calculate unigram bleu score"""
    ref = ngramify(references, n)
    sentence = ngramify(sentence, n)
    sd = {}

    for gram in sentence:
        sd[gram] = sd.get(gram, 0) + 1

    max_dict = {}

    for reference inref:
        this_ref = {}
        for gram in reference:
            this_ref[gram] = this_ref.get(gram, 0) + 1
        for gram in this_ref:
            max_dict[gram] = max(max_dict.get(gram, 0), this_ref[gram])
    
    in_ref = 0

    for gram in sd:
        in_ref += min(max_dict.get(gram, 0), sd[gram])
    
    return w * np.log(in_ref / len(sentence))
