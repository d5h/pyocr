#!/usr/bin/env python

import cv


def binary(img):
    out = cv.CreateImage((img.width, img.height), 8, 1)
    # Use adaptive thresholding with a large blocksize so that dark
    # text on a light background remain dark.
    cv.AdaptiveThreshold(img, out, 255, blockSize=25)
    return out

if __name__ == '__main__':
    import sys
    from show import show

    for a in sys.argv[1:]:
        i = cv.LoadImage(a, cv.CV_LOAD_IMAGE_GRAYSCALE)
        i = binary(i)
        show(i)
        #cv.SaveImage("output.tif", i)
