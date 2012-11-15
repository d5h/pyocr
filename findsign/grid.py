#!/usr/bin/env python


def grid_map(img, grid_rows, grid_cols, f):
    for r in range(grid_rows):
        for c in range(grid_cols):
            gr_start = iround(float(r) / grid_rows * img.rows)
            gr_end = iround(float(r + 1) / grid_rows * img.rows)
            gc_start = iround(float(c) / grid_cols * img.cols)
            gc_end = iround(float(c + 1) / grid_cols * img.cols)
            rect = (gc_start, gr_start, gc_end - gc_start, gr_end - gr_start)
            m = img[gr_start:gr_end, gc_start:gc_end]
            f(m, rect)

def iround(x):
    return int(round(x))

if __name__ == '__main__':
    import cv
    import sys
    mat = cv.LoadImageM(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
    from common.bin import binary
    mat = binary(mat)
    #def print_rect(m, r):
    #    print m[0,0], r
    from entropy import entropy
    data = []
    def print_entropy(m, r):
        data.append(entropy(m))
        print "%s: %s" % (r, data[-1])
    grid_map(mat, 10, 10, print_entropy)
    from common.show import hist
    hist(data)
