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
    mat = cv.CreateMat(40, 25, cv.CV_8U)
    def print_rect(m, r):
        print r
    grid_map(mat, 4, 2, print_rect)
