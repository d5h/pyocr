#!/usr/bin/env python

import cv

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from scipy.stats.stats import pearsonr

from cont import cont_angles, cont_deriv


def image_corr(i, j):
    ai = cont_angles(i)
    aj = cont_angles(j)
    from show import plot
    if len(ai) < len(aj):
        aj = scale_size(aj, ai)
    elif len(aj) < len(ai):
        ai = scale_size(ai, aj)
    di = cont_deriv(ai)
    dj = cont_deriv(aj)
    #plot(ai, 'r', aj, 'b')
    k = find_shift(ai, aj)
    ai = phase_shift(ai, k)
    di = phase_shift(di, k)
    plot(ai, 'r', aj, 'b')
    return pearsonr(ai, aj)[0], pearsonr(di, dj)[0]

def scale_size(a, s):
    na = len(a)
    y = interp1d(range(na), a)
    b = np.empty_like(s)
    nb = len(b)
    for i in range(nb):
        b[i] = y(i * (na - 1.) / (nb - 1))
    return b

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
