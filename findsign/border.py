#!/usr/bin/env python

import cv

from combine.combine_test import CombinedClassifications
from common.bin import binary
from common.score import Classifications
from contours import cont
from contours.cont_test import ContClassifications
from contours.corr import cont_corr
from edge import edges
from entropy import entropy
from common.show import show as showimg


# Auto-generated
angles = [-128.11434100030542, -141.88565899969461, -154.51289230317201, -164.46285922275734, -171.49570794845377, -176.09058887125019, -178.54765013447744, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 178.54765013447744, 176.09058887125019, 171.49570794845377, 164.46285922275734, 154.51289230317201, 141.88565899969461, 128.11434100030542, 115.48710769682801, 105.53714077724266, 98.504292051546258, 93.909411128749838, 91.45234986552255, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 88.547650134477465, 86.090588871250162, 81.495707948453756, 74.462859222757359, 64.512892303172023, 51.885658999694591, 38.114341000305416, 25.487107696827998, 15.537140777242659, 8.5042920515462548, 3.9094111287498374, 1.4523498655225462, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.4523498655225462, -3.9094111287498374, -8.504292051546253, -15.537140777242655, -25.487107696827991, -38.114341000305409, -51.885658999694591, -64.512892303172009, -74.462859222757345, -81.495707948453756, -86.090588871250162, -88.547650134477465, -90.0, -90.0, -90.0, -90.0, -90.0, -90.0, -90.0, -90.0, -90.0, -90.0, -90.0, -90.0, -90.0, -90.0, -90.0, -90.0, -90.0, -91.45234986552255, -93.909411128749838, -98.504292051546244, -105.53714077724266, -115.48710769682798]


class BoxClassifications(Classifications):

    def __init__(self):
        self.boxes = {}

    def add(self, box, correlation):
        self.boxes[box] = correlation

    def certainty(self, box):
        return self.boxes[box]

    def rankings(self, limit=None, ignore_case=True):
        r = sorted(self.boxes, key=self.certainty, reverse=True)
        return self.filter_rankings(r, limit, ignore_case)

def iterseq(s):
    while s is not None:
        yield s
        s = s.h_next()

def show_classifications(classifications, m, limit=10):
    for box in classifications.rankings(limit=limit):
        print box
        cv.Rectangle(m, box[0], box[1], 128, thickness=2)
        showimg(m)

def find_boxes(m, show=False):
    classifications = ContClassifications()
    e = edges(m, show=show)
    # FindContours modifies its source
    e1 = cv.CloneMat(e)
    e2 = cv.CloneMat(e)
    chain_seq = cont.freeman_codes(e1)
    points_seq = cont.contour_points(e2)
    for chain, points in zip(iterseq(chain_seq), iterseq(points_seq)):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        box = (min(xs), min(ys)), (max(xs), max(ys))
        if box[0][0] == box[1][0]:
            continue
        w = box[1][0] - box[0][0]
        h = box[1][1] - box[0][1]
        if w < 20 or h < 40:
            continue
        aspect_ratio = float(h) / w
        if not (0.5 <= aspect_ratio <= 4):
            continue
        print box, aspect_ratio
        a = cont.freeman_to_angles(chain, cont.good_freeman_filter)
        if not a:
            continue
        corr, error = cont_corr(a, angles)
        classifications.add(box, corr, error)
        chain = chain.h_next()
        points = points.h_next()
    if show:
        show_classifications(classifications, m)
    return classifications

def entropy_classifications(boxes, m, ideal=0.083):
    classifications = BoxClassifications()
    binimg = binary(m)
    for b in boxes:
        reg = binimg[b[0][1]:b[1][1], b[0][0]:b[1][0]]
        e = entropy(reg)
        print b, e
        classifications.add(b, 1 - abs(e - ideal))
    return classifications

def find_best_boxes(m):
    cc = CombinedClassifications()
    cc.add(find_boxes(m))
    boxes = cc.rankings()
    cc.add(entropy_classifications(boxes, m))
    return cc

if __name__ == '__main__':
    import sys
    i = cv.LoadImageM(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
    c = find_best_boxes(i)
    show_classifications(c, i, limit=5)
