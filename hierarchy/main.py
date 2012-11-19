#!/usr/bin/env python

import cv

from combine.combine_test import CombinedClassifications, Combiner
from common.bin import binary
from common.score import Classifications
from common.show import show as showimg
from contours import cont
from contours.cont_test import ContClassifications, test as cont_test
from contours.corr import cont_corr
from edge import edges
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
        #reg = self.hierarchy.img[y1:y2, x1:x2]
        #mask = cv.CreateMat(y2 - y1, x2 - x1, cv.CV_8U)
        #cv.Set(mask, 0)
        #cv.DrawContours(mask, freeman_code, external_color=255, hole_color=255,
        #                max_level=0, thickness=cv.CV_FILLED, offset=(-x1, -y1))
        #showimg(mask)
        mask = cv.CreateMat(y2 - y1 + 2, x2 - x1 + 2, cv.CV_8U)
        cv.Set(mask, 0)
        cv.DrawContours(mask, freeman_code, external_color=255, hole_color=0,
                        max_level=-2, thickness=cv.CV_FILLED, offset=(1 - x1, 1 - y1))
        self.char_cls = hierarchy.char_test(mask)
        print self.char_cls.rankings(limit=5)
        showimg(mask)

class Hierarchy(object):

    def __init__(self, img):
        self.objs = []
        self.img = img
        self.binary_img = binary(img)
        self.char_test = Combiner([cont_test, mask_test]).test
        e = edges(img)
        # FindContours modifies the image
        e1 = cv.CloneMat(e)
        e2 = cv.CloneMat(e)
        chain_seq = cont.freeman_codes(e1)
        points_seq = cont.contour_points(e2)
        for chain, points in zip(iterseq(chain_seq), iterseq(points_seq)):
            self.maybe_add_contour(points, chain)

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

class ImgObjClassifications(Classifications):

    def __init__(self):
        pass

    def add(self, obj):
        pass

    def certainty(self, box):
        return

    def rankings(self, limit=None, ignore_case=True):
        r = sorted(self.xyz, key=self.certainty, reverse=True)
        return self.filter_rankings(r, limit, ignore_case)

def iterseq(s):
    while s is not None:
        yield s
        s = s.h_next()

if __name__ == '__main__':
    import sys
    i = cv.LoadImageM(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
    h = Hierarchy(i)
