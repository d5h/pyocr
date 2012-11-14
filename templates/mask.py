#!/usr/bin/env python

import cv
import numpy

from common.show import show as showimg


def make_mask(image, width, height, show=False):
    x, y, w, h = cv.BoundingRect(image)
    image = image[y:y + h, x:x + w]
    result = cv.CreateMat(height, width, cv.CV_8U)
    cv.Resize(image, result)

    b = cv.CreateMat(height, width, cv.CV_8U)
    cv.Threshold(result, b, 128, 1, cv.CV_THRESH_BINARY | cv.CV_THRESH_OTSU)
    result = b

    if show:
        showimg(result)

    return numpy.asarray(result)

def cmp_masks(m1, m2):
    # This is the simplest way to do a element-wise comparison and sum
    # the results
    return numpy.sum(m1 == m2)


if __name__ == '__main__':
    import sys
    i = cv.LoadImageM(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
    make_mask(i, 24, 32, show=True)
