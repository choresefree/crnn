import numpy as np
import torch

from utils.beam_search import beam_decode


def num2str(alphabet, labels):
    """ Convert numeric labels to text labels.
    Args:
        alphabet (str): A dictionary that associates numeric labels with text labels.
        labels (list[list[int]]): List of numeric labels or a separate one.
    Returns:
        texts (list): List of text labels.
    """
    texts = []
    for label in labels:
        text = ''.join(alphabet[i - 1] for i in label)
        texts.append(text)

    return texts


def str2num(alphabet, texts):
    """ Convert text labels to numeric labels.
    Args:
        alphabet (str): Ditto.
        texts (list[str] | str): List of text labels or a separate one.
    Returns:
        labels (list): List of numeric labels.
    """
    if isinstance(texts, str):
        return [alphabet.find(char) + 1 for char in texts]
    labels = []
    for text in texts:
        label = [alphabet.find(char) + 1 for char in text]
        labels.append(label)

    return labels


def greedy_decode(alphabet, sample_pred):
    """ Convert output of single sample to text prediction by greedy search.
    Args:
        alphabet (str): Ditto.
        sample_pred (numpy.array): Output of single sample from network.
    Returns:
        out (str): Prediction of single sample.
    """
    seq = []
    for i in range(sample_pred.shape[0]):
        label = np.argmax(sample_pred[i])
        seq.append(label - 1)
    out = []
    for i in range(len(seq)):
        if len(out) == 0:
            if seq[i] != -1:
                out.append(seq[i])
        else:
            if seq[i] != -1 and seq[i] != seq[i - 1]:
                out.append(seq[i])
    out = ''.join(alphabet[i] for i in out)

    return out


def decode(alphabet, pred, method='greedy'):
    """ Convert output of single sample to text prediction.
    Args:
        alphabet (str): Ditto.
        pred (torch.Tensor): Output of samples in one batch from network.
        method (str): Decoding method.
    Returns:
        out (str): Text prediction of one batch.
    """
    pred = torch.softmax(pred, dim=-1)
    pred = pred.cpu().data.numpy()
    str_pred = []
    for i in range(pred.shape[0]):
        if method == 'greedy':
            str_pred.append(greedy_decode(alphabet, pred[i]))
        elif method == 'beam':
            str_pred.append(beam_decode(pred[i], alphabet, beam_size=5)[0][0])

    return str_pred


def cal_levenshtein(word1, word2):
    """ Calculate the edit distance between two strings. """
    n = len(word1)
    m = len(word2)
    if n * m == 0:
        return n + m
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            left = dp[i - 1][j] + 1
            down = dp[i][j - 1] + 1
            left_down = dp[i - 1][j - 1]
            if word1[i - 1] != word2[j - 1]:
                left_down += 1
            dp[i][j] = min(left, down, left_down)

    return dp[n][m]
