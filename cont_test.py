#!/usr/bin/env python

import os

import cv

from aspect import adjust_aspect
from cont import cont_angles
import cont_table
from corr import cont_corr


class Classifications(object):

    def __init__(self):
        self.correlations = {}
        self.errors = {}
        self.sum_correlations = 0.0
        self.max_error = 0.0

    def add(self, char, correlation, error):
        self.correlations[char] = correlation
        self.errors[char] = error
        self.sum_correlations += abs(correlation)
        self.max_error = max(self.max_error, error)

    def certainty(self, char):
        return abs(self.correlations[char]) * self.error_factor(char) / self.sum_correlations

    def error_factor(self, char):
        return 1 - float(self.errors[char]) / self.max_error

    def rankings(self, limit=None, ignore_case=True):
        r = sorted(self.correlations, key=self.certainty, reverse=True)
        if ignore_case:
            seen = set()
            r = [c for c in r if c not in seen and not seen.add(c)]
        if limit is not None:
            r = r[:limit]
        return r

def test(image, char=None):
    classifications = Classifications()
    image = adjust_aspect(image, binarize=True)
    cont = cont_angles(image)
    for c, a in cont_table.angles.items():
        print c
        corr, error = cont_corr(cont, a)#, show=(c in 'PoDBi'))
        print ("corr(%s, %s) = %.3f; error = %.1f" % (char, c, corr, error))
        classifications.add(c, corr, error)

    return classifications

def score_files(files, ignore_case=True):
    correct = 0
    min_correct_certainty = 1
    ave_correct_certainty = 0
    max_incorrect_certainty = 0
    ave_incorrect_certainty = 0
    incorrect = []
    for f in files:
        char = os.path.basename(f)[0]  # Assume filename starts with char
        i = cv.LoadImageM(f, cv.CV_LOAD_IMAGE_GRAYSCALE)
        classifications = test(i, char=char)
        rankings = classifications.rankings(limit=4)
        guess = rankings[0]
        cert = classifications.certainty(guess)
        case_char = char
        case_guess = guess
        if ignore_case:
            char = char.lower()
            guess = guess.lower()
        if guess == char:
            correct += 1
            min_correct_certainty = min(cert, min_correct_certainty)
            ave_correct_certainty += cert
        else:
            incorrect.append((case_char, case_guess, cert, rankings[1:]))
            max_incorrect_certainty = max(max_incorrect_certainty, cert)
            ave_incorrect_certainty += cert

    n = len(files)
    ave_correct_certainty = float(ave_correct_certainty) / correct
    ave_incorrect_certainty = float(ave_incorrect_certainty) / (n - correct)
    print 'Classified %d/%d correctly (%.2f%%)' % (correct, n, 100. * correct / n)
    print 'Average certainty for correct classification: %.3f (min: %.3f)' % (ave_correct_certainty, min_correct_certainty)
    print 'Average certainty for incorrect classification: %.3f (max: %.3f)' % (ave_incorrect_certainty, max_incorrect_certainty)
    print 'Incorrect classifications:'
    for c, g, r, alt in incorrect:
        print '\tThought %s was %s (certainty: %.3f); alternatives were %s' % (c, g, r, ', '.join(alt))

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
        score_files(args)
