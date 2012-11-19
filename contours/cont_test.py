#!/usr/bin/env python

import os

import cv

from aspect import adjust_aspect
from cont import cont_angles
import cont_table
from corr import cont_corr
from common.score import Classifications, score_files


class ContClassifications(Classifications):

    def __init__(self):
        self.correlations = {}
        self.errors = {}
        self.sum_correlations = 0.0
        self.max_error = 0.0
        self.min_error = float('inf')

    def add(self, obj, correlation, error):
        self.correlations[obj] = correlation
        self.errors[obj] = error
        self.sum_correlations += abs(correlation)
        self.max_error = max(self.max_error, error)
        self.min_error = min(error, self.min_error)

    def certainty(self, obj):
        if self.sum_correlations == 0:
            return 0
        return abs(self.correlations[obj]) * self.error_factor(obj) / self.sum_correlations

    def error_factor(self, obj):
        if self.max_error == self.min_error:
            return 1
        else:
            return 1 - float(self.errors[obj] - self.min_error) / (self.max_error - self.min_error)

    def rankings(self, limit=None, ignore_case=True):
        r = sorted(self.correlations, key=self.certainty, reverse=True)
        return self.filter_rankings(r, limit, ignore_case)

def test(image, char=None):
    classifications = ContClassifications()
    image = adjust_aspect(image, binarize=True)
    cont = cont_angles(image)
    for c, a in cont_table.angles.items():
        #print c
        corr, error = cont_corr(cont, a)#, show=(c in 'PoDBi'))
        #print ("corr(%s, %s) = %.3f; error = %.1f" % (char, c, corr, error))
        classifications.add(c, corr, error)

    return classifications


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    if len(args) == 1:
        i = cv.LoadImageM(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
        classifications = test(i)
        rankings = classifications.rankings(limit=4)
        for r in rankings:
            print "guess = %s; corr = %.3f; error = %.1f; error_factor = %.3f, certainty = %.4f" % (
                r, classifications.correlations[r], classifications.errors[r],
                classifications.error_factor(r), classifications.certainty(r)
                )
        if False:  # plot distributions
            from show import hist
            corrs = classifications.correlations.values()
            errs = [classifications.error_factor(c) for c in classifications.errors]
            certs = [classifications.certainty(c) for c in classifications.correlations]
            hist([corrs, errs, certs], bins=100, label=['correlations', 'error factors', 'certainties'])
    else:
        score_files(args, test)
