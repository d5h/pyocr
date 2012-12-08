#!/usr/bin/env python

from collections import defaultdict

import numpy as np
from scikits import ann

from common.concom import connected_components
from common.score import Classifications, score_files
from common.gen_data import largest_component, params_from_component
from common.data_path import data_path


class KNNTest(object):

    def __init__(self, data_file):
        self.load(data_file)

    def load(self, data_file):
        self.labels = []
        self.label_set = set()
        points = []
        with open(data_file) as fp:
            for line in fp:
                fields = line.split('\t')
                self.labels.append(fields[0])
                self.label_set.add(fields[0])
                points.append([float(f) for f in fields[1:]])
        self.kdtree = ann.kdtree(np.array(points))
        self.n = len(points)
        self.k = 25

    def test(self, image, char=None):
        coms = connected_components(image)
        com = largest_component(coms)
        x = params_from_component(com, with_one=False)
        indexes, errors = self.kdtree.knn(x, self.k)
        scores = {lab: 0.0 for lab in self.label_set}
        for i in indexes[0]:
            scores[self.labels[i]] += 1.0 / self.k
        del scores['']
        return KNNClassifications(scores)

class KNNClassifications(Classifications):

    def __init__(self, scores):
        self._scores = scores

    def certainty(self, char):
        return self._scores[char]

    def rankings(self, limit=None, ignore_case=True):
        r = sorted(self._scores, key=self.certainty, reverse=True)
        return self.filter_rankings(r, limit, ignore_case)


if __name__ == '__main__':
    import sys
    import cv
    args = sys.argv[1:]
    test = KNNTest(data_path)
    if len(args) == 1:
        i = cv.LoadImageM(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
        classifications = test.test(i)
        rankings = classifications.rankings()
        for r in rankings:
            print "guess = %s; certainty = %.4f" % (r, classifications.certainty(r))
    else:
        score_files(args, test.test)
