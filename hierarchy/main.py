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
        self.char = self.char_cls.rankings(limit=1)[0]
        self.char_score = self.char_cls.certainty(self.char)

class Word(object):

    def __init__(self, pos):
        self.chars = []
        self.pos = pos

    def string(self):
        return ''.join([c.char for c in self.chars])

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

        self.compute_char_scores()
        self.group_words()

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

    def compute_char_scores(self):
        self.obj_classifier = CombinedClassifications()
        cls = ImgObjClassifications({obj: obj.char_score for obj in self.objs})
        self.obj_classifier.add(cls)
        cls = ImgObjClassifications({obj: self.char_y_score(obj) for obj in self.objs})
        self.obj_classifier.add(cls)

    def char_y_score(self, obj):
        height = self.img.rows
        m = height * height
        score = 0.0
        y = (obj.bounding_box[1] + obj.bounding_box[3]) / 2.0
        for other in self.objs:
            if obj is other:
                continue
            other_y = (other.bounding_box[1] + other.bounding_box[3]) / 2.0
            score += (m - abs(y - other_y) ** 2) * self.obj_classifier.certainty(other)
        return score / (m * (len(self.objs) - 1))

    def group_words(self, score_threshold=0.5):
        objs = sorted(self.objs, key=lambda c: c.bounding_box[0])
        objs = [b for b in objs if score_threshold < self.obj_classifier.certainty(b)]
        words = []
        word = None
        while objs:
            if word is None:
                c = objs.pop(0)  # would be more efficient to do this backwards
                h = c.bounding_box[3] - c.bounding_box[1]
                y = (c.bounding_box[1] + c.bounding_box[3]) / 2.0
                pos = (c.bounding_box[1], 1)
                word = Word(pos=pos)
                word.chars.append(c)
            for n, d in enumerate(objs):
                ylim = max(h, d.bounding_box[3] - d.bounding_box[1]) / 2.0
                if abs((d.bounding_box[1] + d.bounding_box[3]) / 2.0 - y) < ylim:
                    if (c.bounding_box[2] - c.bounding_box[0] + d.bounding_box[2] - d.bounding_box[0]) / 4.0 \
                           < (d.bounding_box[0] - c.bounding_box[2]):
                        words.append(word)
                        pos = (pos[0], pos[1] + 1)
                        word = Word(pos)
                    c = d
                    h = c.bounding_box[3] - c.bounding_box[1]
                    y = (c.bounding_box[1] + c.bounding_box[3]) / 2.0
                    word.chars.append(c)
                    del objs[n]
                    break
            else:
                words.append(word)
                word = None
        if word:
            words.append(word)
        self.words = sorted(words, key=lambda w: w.pos)

    def output(self):
        line = self.words[0].pos[0]
        for w in self.words:
            if w.pos[0] != line:
                line = w.pos[0]
                print
            print w.string(),

class ImgObjClassifications(Classifications):

    def __init__(self, init=None):
        if init:
            self._objs = init
        else:
            self._objs = {}

    def add(self, obj, score):
        self._objs[obj] = score

    def certainty(self, obj):
        return self._objs[obj]

    def rankings(self, limit=None, ignore_case=False):
        r = sorted(self._objs, key=self.certainty, reverse=True)
        return self.filter_rankings(r, limit, ignore_case=False)

def iterseq(s):
    while s is not None:
        yield s
        s = s.h_next()

if __name__ == '__main__':
    i = cv.LoadImageM(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
    h = Hierarchy(i)
    h.output()
