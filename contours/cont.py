#!/usr/bin/env python

import cv
import numpy as np
from scipy.signal import gaussian

from common.show import plot, show


sqrt_half = 0.70710678118654746

freeman_to_vector = {
    0: np.array([0, 1]),
    1: np.array([sqrt_half, sqrt_half]),
    2: np.array([1, 0]),
    3: np.array([sqrt_half, -sqrt_half]),
    4: np.array([0, -1]),
    5: np.array([-sqrt_half, -sqrt_half]),
    6: np.array([-1, 0]),
    7: np.array([-sqrt_half, sqrt_half])
    }

good_freeman_filter = gaussian(11, 3)

def freeman_codes(image, flat=False):
    retr = cv.CV_RETR_LIST if flat else cv.CV_RETR_TREE
    return cv.FindContours(image, cv.CreateMemStorage(), retr, cv.CV_CHAIN_CODE)

def contour_points(image, flat=False):
    # Returns the points associated with the Freeman codes.
    retr = cv.CV_RETR_LIST if flat else cv.CV_RETR_TREE
    return cv.FindContours(image, cv.CreateMemStorage(), retr)

def cont_angles(image):
    seq = freeman_codes(image)
    return freeman_to_angles(seq, gaussian(11, 3))

def freeman_to_angles(fr, conv_window=None):
    vectors = [freeman_to_vector[f] for f in fr]
    if conv_window is not None:
        vectors = convolve_window_wrap(vectors, conv_window)

    return [angle([1, 0], vectors[n]) for n in range(len(vectors))]

def angle(x, y):
    """Signed angle using perp dot product over dot product"""
    return np.degrees(np.arctan2(x[0] * y[1] - x[1] * y[0], x[0] * y[0] + x[1] * y[1]))

def convolve_window_wrap(a, w):
    """Might normalize w"""

    assert len(w) % 2 == 1
    s = sum(w)
    if s != 1:
        w = w / float(s)  # Note, /= doesn't cast to float :\

    m = len(w) / 2
    c = np.empty_like(a, dtype=float)
    n = len(a)
    for i in range(n):
        c[i] = sum([w[j] * a[(i + j - m) % n] for j in range(len(w))])

    return c

if __name__ == '__main__':
    import sys
    image = cv.LoadImageM(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
    show_cont_change(image)
