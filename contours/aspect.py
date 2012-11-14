#!/usr/bin/env python

import cv

from common.show import show as showimg


def adjust_aspect(img, a=1.5, show=False, binarize=False):
    """Resizes img such that the object defined by ON pixels within
    has aspect ratio (height / width) a."""

    if show:
        showimg(img)
    min_rect = cv.BoundingRect(img)
    wr, hr = min_rect[2:]
    w, h = cv.GetSize(img)
    orig_aspect = float(hr) / wr
    if orig_aspect < a:
        new_w = w
        new_h = int(round(a * h / orig_aspect))
    else:
        new_w = int(round(orig_aspect * w / a))
        new_h = h
    #print w, h, new_w, new_h
    result = cv.CreateMat(new_h, new_w, cv.CV_8U)
    cv.Resize(img, result)

    if binarize:
        b = cv.CreateImage((result.width, result.height), 8, 1)
        # Threshold of 128 should work if the original image was
        # binary.
        cv.Threshold(result, b, 128, 255, cv.CV_THRESH_BINARY | cv.CV_THRESH_OTSU)
        result = b

    if show:
        print "Original: %.2f" % orig_aspect
        showimg(result)

    return result


if __name__ == '__main__':
    import sys
    i = cv.LoadImageM(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
    adjust_aspect(i, show=True, binarize=True)
