#!/usr/bin/env python

import cv

from mask import cmp_masks, make_mask
import mask_table
from common.score import Classifications, score_files


class MaskClassifications(Classifications):

    def __init__(self):
        self._sums = {}
        self._total = 0.0

    def add(self, char, sum):
        self._sums[char] = sum
        self._total += sum

    def certainty(self, char):
        c = self._sums[char] / self._total
        # Note that simply punishing these letters increases accuracy
        # from ~69% to 75% in one test.
        #if char in 'Iil':
        #    c /= 2.0
        return c

    def rankings(self, limit=None, ignore_case=True):
        r = sorted(self._sums, key=self.certainty, reverse=True)
        return self.filter_rankings(r, limit, ignore_case)

def test(image, char=None):
    classifications = MaskClassifications()
    m1 = make_mask(image, width=mask_table.width, height=mask_table.height)
    for c, m2 in mask_table.masks.items():
        s = cmp_masks(m1, m2)
        classifications.add(c, s)

    return classifications


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    if len(args) == 1:
        i = cv.LoadImageM(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
        classifications = test(i)
        rankings = classifications.rankings(limit=4)
        for r in rankings:
            print "guess = %s; certainty = %.4f" % (r, classifications.certainty(r))
    else:
        score_files(args, test)
