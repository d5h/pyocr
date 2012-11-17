#!/usr/bin/env python

import cv

from common.show import show as showimg


def edges(m, show=False):
    e = cv.CreateMat(m.rows, m.cols, cv.CV_8U)
    cv.Smooth(m, e, cv.CV_GAUSSIAN, 3, 3)
    cv.Canny(e, e, 30, 90)
    if show:
        showimg(e)
    return e

if __name__ == '__main__':
    import sys
    i = cv.LoadImageM(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
    edges(i, True)
