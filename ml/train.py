#!/usr/bin/env python

from pprint import pformat

import numpy as np
from numpy.linalg import lstsq


class Trainer(object):

    def __init__(self, data):
        self.data = []
        self.classes = set()
        self.models = {}
        self.load_data(data)

    def load_data(self, filename):
        with open(filename) as f:
            for line in f:
                cls, rest = line.split(',', 1)
                xs = [float(x) for x in rest.split(',')]
                if cls:
                    self.classes.add(cls)
                self.data.append((xs, cls))

    def train(self):
        for cls in self.classes:
            # Use linear regression to initialize weights
            x, y = self.get_classified_data(cls)
            w = lstsq(x, y)[0]
            self.models[cls] = tuple(w)

    def get_classified_data(self, for_cls):
        """Returns the matrix X and vector Y such that the row X[i] is
        classified as Y[i]."""

        rows = len(self.data)
        cols = len(self.data[0][0]) + 1
        x = np.empty((rows, cols))
        y = np.empty(rows)
        for i, (xs, cls) in enumerate(self.data):
            x[i][0] = 1
            x[i][1:] = xs
            y[i] = 2 * (cls == for_cls) - 1

        return x, y

    def output(self, filename):
        with open(filename, 'w') as f:
            self.output_fp(f)

    def output_fp(self, fp):
        fp.write("# Auto-generated\n")
        fp.write("models = %s\n" % pformat(self.models))


if __name__ == '__main__':
    import sys
    t = Trainer(sys.argv[1])
    t.train()
    t.output('hypotheses.py')
