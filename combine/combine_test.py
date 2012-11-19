#!/usr/bin/env python

from collections import defaultdict

import cv

from contours.cont_test import test as cont_test
from templates.mask_test import test as mask_test
from common.score import Classifications, score_files


class CombinedClassifications(Classifications):

    def __init__(self):
        self._certainties = defaultdict(lambda: 1.0)

    def add(self, sub_cls):
        sub_rankings = sub_cls.rankings()
        max_cert = float(sub_cls.certainty(sub_rankings[0]))
        min_cert = sub_cls.certainty(sub_rankings[-1])
        if max_cert != min_cert:
            normalized_certainties = [(sub_cls.certainty(obj) - min_cert) / (max_cert - min_cert) for obj in sub_rankings]
        else:
            normalized_certainties = [1] * len(sub_rankings)
        for n, obj in enumerate(sub_rankings):
            self._certainties[obj] *= normalized_certainties[n]

    def certainty(self, obj):
        return self._certainties[obj]

    def rankings(self, limit=None, ignore_case=True):
        r = sorted(self._certainties, key=self.certainty, reverse=True)
        return self.filter_rankings(r, limit, ignore_case)

class Combiner(object):

    def __init__(self, tests):
        self._tests = tests

    def test(self, image, char=None):
        classifications = CombinedClassifications()
        for test in self._tests:
            classifications.add(test(image, char=char))

        return classifications


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    c = Combiner([cont_test, mask_test])
    if len(args) == 1:
        i = cv.LoadImageM(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
        classifications = c.test(i)
        rankings = classifications.rankings(limit=4)
        for r in rankings:
            print "guess = %s; certainty = %.4f" % (r, classifications.certainty(r))
    else:
        score_files(args, c.test)
