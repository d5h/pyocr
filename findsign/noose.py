#!/usr/bin/env python

def tighten(m, f, step=5):
    yt, yb = 0, m.rows
    xl, xr = 0, m.cols

    score = f(m)
    while step < yb - yt and step < xr - xl:
        top_score = f(m[yt + step:yb, xl:xr])
        bottom_score = f(m[yt:yb - step, xl:xr])
        left_score = f(m[yt:yb, xl + step:xr])
        right_score = f(m[yt:yb, xl:xr - step])

        ranked_scores = sorted([score, top_score, bottom_score, left_score, right_score], reverse=True)
        best = ranked_scores[0]
        print xl, yt, xr - xl, yb - yt, ranked_scores
        if best == score:
            return xl, yt, xr - xl, yb - yt
        if best == top_score:
            yt += step
        elif best == bottom_score:
            yb -= step
        elif best == left_score:
            xl += step
        else:
            xr -= step

if __name__ == '__main__':
    import cv
    import sys
    from entropy import entropy
    ideal = float(sys.argv[2])
    m = cv.LoadImageM(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
    def score(m):
        e = entropy(m)
        return 1.0 / abs(e - ideal)
    r = tighten(m, score)
    if r:
        from common.show import show
        print r
        show(cv.GetSubRect(m, r))
    else:
        print "Degenerate"
