#!/usr/bin/env python

import cv


def binary(img, invert=False):
    thresh_type = cv.CV_THRESH_BINARY if not invert else cv.CV_THRESH_BINARY_INV
    out = cv.CreateMat(img.height, img.width, cv.CV_8U)
    # Use adaptive thresholding with a large blocksize so that dark
    # text on a light background remain dark.
    cv.AdaptiveThreshold(img, out, 255, blockSize=25, thresholdType=thresh_type)
    return out

if __name__ == '__main__':
    import sys
    from show import show

    for a in sys.argv[1:]:
        i = cv.LoadImage(a, cv.CV_LOAD_IMAGE_GRAYSCALE)
        i = binary(i, invert=True)
        show(i)
        #cv.SaveImage("output.tif", i)
