#!/usr/bin/env python

import cv

import numpy as np
from scipy.signal import fftconvolve
from scipy.stats.stats import pearsonr

from cont import cont_angles


def image_corr(i, j):
    ai = cont_angles(i)
    aj = cont_angles(j)
    return cont_corr(ai, aj)

def cont_corr(ai, aj, show=False):
    if len(ai) == 0 or len(aj) == 0:
        return 0, float('inf')
    if len(ai) < len(aj):
        aj = scale_size(aj, ai)
    elif len(aj) < len(ai):
        ai = scale_size(ai, aj)
    k = find_shift(ai, aj)
    ai = phase_shift(ai, k)
    if show:
        from show import plot
        plot(ai, 'r', aj, 'b')
    return pearsonr(ai, aj)[0], sum(abs(ai - aj))  # error

def scale_size(a, s):
    na = len(a)
    ns = len(s)
    xs = [i * (na - 1.) / (ns - 1) for i in range(ns)]
    return np.interp(xs, range(na), a)

def find_shift(a, b):
    c = fftconvolve(np.array(a), np.array(b[::-1]))  # Reverse for cross-correlation
    return np.argmax(c) - len(b)

def phase_shift(a, k):
    n = len(a)
    s = np.concatenate((a[k:], a[:k]))
    assert len(s) == n
    return s

if __name__ == '__main__':
    import sys
    i = cv.LoadImageM(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
    j = cv.LoadImageM(sys.argv[2], cv.CV_LOAD_IMAGE_GRAYSCALE)
    print image_corr(i, j)
