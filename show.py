#!/usr/bin/env python

import cv


window = "Demo"
cv.NamedWindow(window, cv.CV_WINDOW_AUTOSIZE)


def plot(*args, **kwargs):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(*args, **kwargs)
    plt.show()

def hist(*args, **kwargs):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(*args, **kwargs)
    if isinstance(args[0], list) and 1 < len(args[0]):
        ax.legend()
    plt.show()

def show(img):
    cv.ShowImage(window, img)
    cv.WaitKey()


if __name__ == '__main__':
    import sys

    for a in sys.argv[1:]:
        i = cv.LoadImage(a)
        show(i)
