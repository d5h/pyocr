#!/usr/bin/env python

import cv

import numpy as np
from scipy.interpolate import interp1d
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
    plot(ai)
    plot(aj)
    return pearsonr(ai, aj)[0], pearsonr(di, dj)[0]

def scale_size(a, s):
    na = len(a)
    y = interp1d(range(na), a)
    b = np.empty_like(s)
    nb = len(b)
    for i in range(nb):
        b[i] = y(i * (na - 1.) / (nb - 1))
    return b

if __name__ == '__main__':
    import sys
    i = cv.LoadImageM(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
    j = cv.LoadImageM(sys.argv[2], cv.CV_LOAD_IMAGE_GRAYSCALE)
    print image_corr(i, j)
