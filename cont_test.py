#!/usr/bin/env python

import os

import cv

from cont import cont_angles
import cont_table
from corr import cont_corr


def test(image):
    cont = cont_angles(image)
    guess = ''
    max_corr = 0
    for c, a in cont_table.angles.items():
        corr = cont_corr(cont, a)
        if max_corr < corr:
            guess = c
            max_corr = corr

    return guess, max_corr

def score_files(files, ignore_case=True):
    correct = 0
    min_correct_corr = 1
    ave_correct_corr = 0
    max_incorrect_corr = 0
    ave_incorrect_corr = 0
    incorrect = {}
    for f in files:
        char = os.path.basename(f)[0]  # Assume filename starts with char
        i = cv.LoadImageM(f, cv.CV_LOAD_IMAGE_GRAYSCALE)
        guess, corr = test(i)
        if ignore_case:
            char = char.lower()
            guess = guess.lower()
        if guess == char:
            correct += 1
            min_correct_corr = min(corr, min_correct_corr)
            ave_correct_corr += corr
        else:
            incorrect[char] = guess, corr
            max_incorrect_corr = max(max_incorrect_corr, corr)
            ave_incorrect_corr += corr

    n = len(files)
    ave_correct_corr = float(ave_correct_corr) / n
    ave_incorrect_corr = float(ave_incorrect_corr) / n
    print 'Classified %d/%d correctly (%.2f%%)' % (correct, n, 100. * correct / n)
    print 'Average correlation for correct classification: %.3f (min: %.3f)' % (ave_correct_corr, min_correct_corr)
    print 'Average correlation for incorrect classification: %.3f (max: %.3f)' % (ave_incorrect_corr, max_incorrect_corr)
    print 'Incorrect classifications:'
    for c, (g, r) in incorrect.items():
        print '\tThought %s was %s (corr: %.3f)' % (c, g, r)

if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    if len(args) == 1:
        i = cv.LoadImageM(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
        print test(i)
    else:
        score_files(args)
