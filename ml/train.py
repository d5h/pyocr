#!/usr/bin/env python

from itertools import combinations_with_replacement
from pprint import pformat
from random import randrange

import numpy as np
from numpy.linalg import lstsq
from scipy.optimize.optimize import fmin_bfgs


class Trainer(object):

    def __init__(self, data, alpha=0.01, polynomial_transform_order=1):
        """alpha is the regularization coefficient."""
        self.data = []
        self.alpha = alpha
        self.classes = set()
        self.models = {}
        self.polynomial_transform_order = polynomial_transform_order
        self.load_data(data)

    def load_data(self, filename):
        with open(filename) as f:
            for line in f:
                cls, rest = line.split(',', 1)
                xs = [float(x) for x in rest.split(',')]
                if cls:
                    self.classes.add(cls)
                self.data.append((xs, cls))

    @staticmethod
    def sigmoid(s):
        return np.exp(s) / (1 + np.exp(s))

    def train(self):
        for cls in self.classes:
            print "Training on", cls
            x, y = self.get_classified_data(cls)
            z0 = self.get_transformed_data(x[0], self.polynomial_transform_order)
            w = np.random.random_sample(len(z0))
            w = fmin_bfgs(self.make_error(x, y), w, fprime=self.make_error_gradient(x, y))
            self.models[cls] = w

    @staticmethod
    def get_transformed_data(x, poly_order):
        if poly_order <= 1:
            z = x
        else:
            z = [a * b for (a, b) in combinations_with_replacement(x, poly_order)]
        return np.array(z)

    def make_error(self, xs, ys):
        def error(w):
            e = 0.0
            for x, y in zip(xs, ys):
                z = self.get_transformed_data(x, self.polynomial_transform_order)
                s = np.exp(-y * np.dot(z, w))
                e += np.log(1 + s)
            return e / len(ys) + self.alpha * np.dot(w, w) / 2

        return error

    def make_error_gradient(self, xs, ys):
        def error_gradient(w):
            e = np.zeros(len(w))
            for x, y in zip(xs, ys):
                z = self.get_transformed_data(x, self.polynomial_transform_order)
                s = 1 + np.exp(y * np.dot(z, w))
                e -= y * z / s
            e /= len(ys)
            e += self.alpha * w
            return e

        return error_gradient

    def get_classified_data(self, for_cls):
        """Returns the matrix X and vector Y such that the row X[i] is
        classified as Y[i]."""

        rows = 0
        for xs, cls in self.data:
            if not cls or cls == for_cls:
                rows += 1

        cols = len(self.data[0][0]) + 1
        x = np.empty((rows, cols))
        y = np.empty(rows)

        i = 0
        for xs, cls in self.data:
            if not cls or cls == for_cls:
                x[i][0] = 1
                x[i][1:] = xs
                y[i] = 2 * (cls == for_cls) - 1
                i += 1

        return x, y

    def output(self, filename):
        with open(filename, 'w') as f:
            self.output_fp(f)

    def output_fp(self, fp):
        fp.write("# Auto-generated\n")
        fp.write("from numpy import array\n")
        fp.write("models = %s\n" % pformat(self.models))
        fp.write("polynomial_transform_order = %d\n" % self.polynomial_transform_order)


if __name__ == '__main__':
    import sys
    t = Trainer(sys.argv[1])
    t.train()
    t.output('hypotheses.py')
