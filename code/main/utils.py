import unicodedata
import numpy as np


def mean(arr, normalize=False):
    r = sum(arr) / len(arr)
    if normalize:
        return r / (np.linalg.norm(r) + 1e-12)
    return r


def _count_cjk_chars(string):
    return sum(unicodedata.east_asian_width(c) in "FW" for c in string)


def ljust(string, width, fillbyte=" "):
    return string.ljust(width - _count_cjk_chars(string), fillbyte)
