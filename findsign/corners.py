#!/usr/bin/env python

import cv

from common.show import show as showimg


def corners(m, template_width=30, template_height=40,
            line_thickness=1, line_spacing=3, num_lines=3,
            show=False):

    template = cv.CreateMat(template_height, template_width, cv.CV_8U)
    cv.Set(template, 0)

    draw_lines(template, cv.GetRow, num_lines, line_thickness, line_spacing)
    draw_lines(template, cv.GetCol, num_lines, line_thickness, line_spacing)

    if show:
        showimg(template)

    res_top_left = cv.CreateMat(m.rows - template.rows + 1, m.cols - template.cols + 1, cv.CV_32F)
    cv.MatchTemplate(m, template, res_top_left, cv.CV_TM_SQDIFF_NORMED)
    print res_top_left.type
    print cv.MinMaxLoc(res_top_left)
    _, _, (mx, my), _ = cv.MinMaxLoc(res_top_left)
    #_, _, _, (mx, my) = cv.MinMaxLoc(res_top_left)
    if show:
        # kludge
        cv.Copy(template, m[my:my + template.rows,mx:mx + template.cols])
        showimg(m)
        showimg(res_top_left)

def draw_lines(m, getter, num, thickness, spacing):
    x = 0
    for n in range(num):
        for t in range(thickness):
            cv.Set(getter(m, x), 255)
            x += 1
        x += spacing

if __name__ == '__main__':
    import sys
    from edge import edges
    i = cv.LoadImageM(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
    i = edges(i, show=True)
    corners(i, show=True)
