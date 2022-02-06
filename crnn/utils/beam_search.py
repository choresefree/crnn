""" Copy from https://blog.csdn.net/weixin_42615068/article/details/93767781 Beam search is an improved algorithm for
happy search. Compared with green search, it expands the search space, but it is far less than exhausting the search
space of index level. It is a compromise between the two
"""
import numpy as np


def remove_blank(labels, blank=0):
    new_labels = []
    # Merge the same labels.
    previous = None
    for label in labels:
        if label != previous:
            new_labels.append(label)
            previous = label
    # Delete blank labels.
    new_labels = [new_label for new_label in new_labels if new_label != blank]

    return new_labels


def beam_decode(y, alphabet, beam_size=10):
    """ Realization of beam search.
    Args:
        y (numpy.array): The probability of each prediction character at each time (t, num_class).
        alphabet (str): A dictionary that associates numeric labels with text labels.
        beam_size (int): Number of predicted roads to keep.
    """
    t, num_class = y.shape
    log_y = np.log(y)  # Prevent lower overflow
    beam = [([], 0)]
    for t in range(t):
        new_beam = []
        for prefix, score in beam:
            for i in range(num_class):
                new_prefix = prefix + [i]
                new_score = score + log_y[t, i]
                new_beam.append((new_prefix, new_score))
        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_size]  # Front row road with higher probability

    pres, scores = zip(*beam)
    pres = [''.join([alphabet[i - 1] for i in remove_blank(pre)]) for pre in pres]

    return list(zip(pres, scores))
