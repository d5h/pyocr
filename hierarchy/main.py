#!/usr/bin/env python

import sys

import cv
import numpy as np

from combine.combine_test import CombinedClassifications, Combiner
from common.bin import binary
from common.score import Classifications
from common.show import show as showimg
from contours import cont
from contours.cont_test import ContClassifications, test as cont_test
from contours.corr import cont_corr
from sign_angles import sign_angles
from templates.mask_test import test as mask_test


class ImgObj(object):

    def __init__(self, contour_points, freeman_code, hierarchy, bounding_box):
        self.hierarchy = hierarchy
        self.contour_points = contour_points
        self.freeman_code = freeman_code
        self.bounding_box = bounding_box
        self.nested = []

        x1, y1, x2, y2 = bounding_box
        mask = cv.CreateMat(y2 - y1 + 2, x2 - x1 + 2, cv.CV_8U)
        cv.Set(mask, 0)
        cv.Copy(self.hierarchy.binary_img[y1:y2, x1:x2], mask[1:-1,1:-1])
        self.char_cls = hierarchy.char_test(mask)
        self.char_score = self.char_cls.certainty(self.char_cls.rankings(limit=1)[0])

    def compute_char_scores(self):
        self.char_y_score = char_y_test(self, self.hierarchy)

class Hierarchy(object):

    def __init__(self, img):
        self.objs = []
        self.img = img
        self.binary_img = binary(img, invert=True)
        self.char_test = Combiner([cont_test, mask_test]).test
        b = self.binary_img
        # FindContours modifies the image
        b1 = cv.CloneMat(b)
        b2 = cv.CloneMat(b)
        chain_seq = cont.freeman_codes(b1)
        points_seq = cont.contour_points(b2)
        for chain, points in zip(iterseq(chain_seq), iterseq(points_seq)):
            self.maybe_add_contour(points, chain)

        for obj in self.objs:
            obj.compute_char_scores()
        self.sort_objects()

    def maybe_add_contour(self, points, chain, parent=None):
        if len(points) < 4 or len(chain) < 15:
            return
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
        w = xmax - xmin
        h = ymax - ymin
        if w < 5 or h < 10:
            return
        aspect_ratio = float(h) / w
        if not (0.5 <= aspect_ratio <= 4):
            return

        obj = ImgObj(contour_points=points, freeman_code=chain, hierarchy=self,
                     bounding_box=(xmin, ymin, xmax, ymax))
        self.objs.append(obj)
        if parent:
            parent.nested.append(obj)

        # Note that we don't check nested objects of those that are
        # rejected by maybe_add_contour
        points_child = points.v_next()
        if points_child is not None:
            self.maybe_add_contour(points_child, chain.v_next(), obj)

    def sort_objects(self):
        # Snap y coordinate to adjust for tilted / uneven characters.
        # We use the median height of objects to determine the snap
        # distance.  Hopefully characters fall in the median size
        # range.
        heights = [c.bounding_box[3] - c.bounding_box[1] for c in self.objs]
        med_height = np.median(heights)
        snap = med_height / 2.0

        def comparator(a, b):
            ax = a.bounding_box[0]
            ay = int(round((a.bounding_box[1] + a.bounding_box[3]) / snap))
            bx = b.bounding_box[0]
            by = int(round((b.bounding_box[1] + b.bounding_box[3]) / snap))
            return cmp(ay, by) or cmp(ax, bx)

        self.objs.sort(cmp=comparator)

    def output(self):
        for obj in self.objs:
            c = obj.char_cls.rankings(limit=1)[0]
            print c, obj.char_score, obj.char_y_score

class ImgObjClassifications(Classifications):

    def __init__(self):
        self._objs = {}

    def add(self, obj, score):
        self._objs[obj] = score

    def certainty(self, obj):
        return self._objs[obj]

    def rankings(self, limit=None, ignore_case=False):
        r = sorted(self._objs, key=self.certainty, reverse=True)
        return self.filter_rankings(r, limit, ignore_case=False)

def char_y_test(obj, hierarchy):
    height = hierarchy.img.rows
    m = height * height
    score = 0.0
    y = (obj.bounding_box[1] + obj.bounding_box[3]) / 2.0
    for other in hierarchy.objs:
        if obj is other:
            continue
        other_y = (other.bounding_box[1] + other.bounding_box[3]) / 2.0
        score += (m - abs(y - other_y) ** 2) * other.char_score
    return score / (m * (len(hierarchy.objs) - 1))

def iterseq(s):
    while s is not None:
        yield s
        s = s.h_next()

if __name__ == '__main__':
    i = cv.LoadImageM(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
    h = Hierarchy(i)
    h.output()
