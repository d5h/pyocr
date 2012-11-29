#!/usr/bin/env python

from pprint import pformat
from random import randrange

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

    def train(self, epochs=100):
        for cls in self.classes:
            # Use linear regression to initialize weights
            x, y = self.get_classified_data(cls)
            w = lstsq(x, y)[0]

            n = y.shape[0]
            # Find positive classifications.  Because we have so many
            # negatives, we make sure that the positives classified
            # correctly.  Otherwise they do not contribute enough to
            # the error.
            positive_rows = [i for i in range(n) if 0 < y[i]]
            for p in positive_rows:
                w = self.adjust(w, x[p], y[p])

            error = self.calc_error(w, x, y)
            for _ in range(epochs):
                i = randrange(n)
                v = self.adjust(w, x[i], y[i])
                v_error = self.calc_error(v, x, y)
                if v_error < error:
                    w = v
                    error = v_error

            self.models[cls] = tuple(w)

    def adjust(self, w, x, y):
        if y * np.dot(w, x) < 0:
            return w + y * x
        return w

    def calc_error(self, w, x, y):
        error = 0
        n = y.shape[0]
        for i in range(n):
            if y[i] * np.dot(w, x[i]) < 0:
                if 0 < y[i]:
                    error += n
                else:
                    error += 1
        return error

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
