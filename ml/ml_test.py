#!/usr/bin/env python

from math import copysign

import cv
from numpy import dot, mean

from common.concom import connected_components
from common.score import Classifications, score_files
from gen_data import largest_component, params_from_component
from hypotheses import models, polynomial_transform_order
from train import Trainer


class MLClassifications(Classifications):

    def __init__(self):
        self._classes = {}
        self._error = 0.1  # FIXME: get this from test data

    def add(self, char, cls):
        """cls should be +1 or -1."""
        self._classes[char] = cls

    def certainty(self, char):
        return self._classes[char]

    def rankings(self, limit=None, ignore_case=True):
        r = sorted(self._classes, key=self.certainty, reverse=True)
        return self.filter_rankings(r, limit, ignore_case)

def test(image, char=None):
    classifications = MLClassifications()
    coms = connected_components(image)
    com = largest_component(coms)
    xs = params_from_component(com, with_one=True)
    for c, ws in models.items():
        s = dot(ws, Trainer.get_transformed_data(xs, polynomial_transform_order))
        print c, s
        classifications.add(c, s)

    return classifications

def sign(y):
    return copysign(1, y)


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    if len(args) == 1:
        i = cv.LoadImageM(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
        classifications = test(i)
        rankings = classifications.rankings()
        certainties = [classifications.certainty(r) for r in rankings]
        for n, r in enumerate(rankings):
            print "guess = %s; certainty = %.4f" % (r, certainties[n])
        print "mean certainty = %.4f" % mean(certainties)
        positive = [c for c in certainties if 0 < c]
        print "num positive = %d (%d%%)" % (len(positive), round(100.0 * len(positive) / len(certainties)))
    else:
        score_files(args, test)
