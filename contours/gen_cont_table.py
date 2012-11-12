#!/usr/bin/env python

import os

import cv

from aspect import adjust_aspect
from cont import cont_angles
from corr import cont_corr


class ContTable(object):

    def __init__(self):
        self._cont = {}

    def add(self, char, image):
        image = adjust_aspect(image, binarize=True)
        self._cont[char] = cont_angles(image)

    def save(self, filename='cont_table.py'):
        with open(filename, 'w') as out:
            out.write('# Auto-generated\n')
            out.write('from numpy import array\n')
            out.write('angles = {\n')
            for c, a in self._cont.items():
                out.write('    %r: %r,\n' % (c, a))
            out.write('}\n')

    def from_files(self, files):
        for f in files:
            char = os.path.basename(f)[0]  # Assume filename starts with character
            image = cv.LoadImageM(f, cv.CV_LOAD_IMAGE_GRAYSCALE)
            self.add(char, image)


if __name__ == '__main__':
    import sys
    ct = ContTable()
    ct.from_files(sys.argv[1:])
    ct.save()
