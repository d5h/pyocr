#!/usr/bin/env python

from collections import defaultdict

import cv

from contours.cont_test import test as cont_test
from templates.mask_test import test as mask_test
from common.score import Classifications, score_files


class CombinedClassifications(Classifications):

    def __init__(self, certainties):
        self._certainties = certainties

    def certainty(self, char):
        return self._certainties[char]

    def rankings(self, limit=None, ignore_case=True):
        r = sorted(self._certainties, key=self.certainty, reverse=True)
        return self.filter_rankings(r, limit, ignore_case)

class Combiner(object):

    def __init__(self, tests):
        self._tests = tests

    def test(self, image, char=None):
        certainties = defaultdict(lambda: 1.0)
        for test in self._tests:
            sub_cls = test(image, char=char)
            sub_rankings = sub_cls.rankings()
            max_cert = float(sub_cls.certainty(sub_rankings[0]))
            min_cert = sub_cls.certainty(sub_rankings[-1])
            normalized_certainties = [(sub_cls.certainty(char) - min_cert) / (max_cert - min_cert) for char in sub_rankings]
            for n, char in enumerate(sub_rankings):
                certainties[char] *= normalized_certainties[n]

        return CombinedClassifications(certainties)


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
