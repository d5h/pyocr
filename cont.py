#!/usr/bin/env python

import cv
import numpy as np
from scipy.signal import gaussian

from show import plot, show


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

def show_cont_change(image):
    seq = cv.FindContours(image, cv.CreateMemStorage(), cv.CV_RETR_TREE, cv.CV_CHAIN_CODE)
    angles = freeman_to_angles(seq, gaussian(11, 3))
    diff = [angle_diff(angles[n], angles[n - 1]) for n in range(len(angles))]
    cv.Set(image, 0)
    cv.DrawContours(image, seq, cv.Scalar(255), cv.Scalar(255), 0)

    #show(image)
    #plot(seq)
    plot(angles)
    plot(diff)
    del seq   # associated storage is also released

def freeman_to_angles(fr, conv_window=None):
    vectors = [freeman_to_vector[f] for f in fr]
    if conv_window is not None:
        vectors = convolve_window_wrap(vectors, conv_window)

    return [angle([1, 0], vectors[n]) for n in range(len(vectors))]

def angle(x, y):
    """Signed angle using perp dot product over dot product"""
    return np.degrees(np.arctan2(x[0] * y[1] - x[1] * y[0], x[0] * y[0] + x[1] * y[1]))

def angle_diff(a, b):
    """Angle difference with abs value less than 180"""
    d = b - a
    if 180 < d:
        d -= 360
    elif d < -180:
        d += 360
    return d

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
