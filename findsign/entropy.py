#!/usr/bin/env python

import cv
import numpy as np


def entropy(img):
    """Only works with single-channel, CV_8U images."""

    assert img.channels == 1
    assert img.type == cv.CV_8U

    e = 0
    for r in range(img.rows):
        vals = [bool(img[r, c]) for c in range(img.cols)]
        e += np.sum(np.diff(vals))

    for c in range(img.cols):
        vals = [bool(img[r, c]) for r in range(img.rows)]
        e += np.sum(np.diff(vals))

    return float(e) / (2 * img.rows * img.cols - img.rows - img.cols)

if __name__ == '__main__':
    import sys
    mat = cv.LoadImageM(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
    print entropy(mat)
