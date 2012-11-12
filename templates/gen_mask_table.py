#!/usr/bin/env python

import os

import cv

from mask import make_mask


class MaskTable(object):

    def __init__(self, width=24, height=32):
        self._masks = {}
        self._w = width
        self._h = height

    def add(self, char, image):
        mask = make_mask(image, width=self._w, height=self._h)
        self._masks[char] = mask

    def save(self, filename='mask_table.py'):
        with open(filename, 'w') as out:
            out.write('# Auto-generated\n')
            out.write('from numpy import array, uint8\n')
            out.write('width, height = %d, %d\n' % (self._w, self._h))
            out.write('masks = {\n')
            for c, m in self._masks.items():
                out.write('    %r: %r,\n' % (c, m))
            out.write('}\n')

    def from_files(self, files):
        for f in files:
            char = os.path.basename(f)[0]  # Assume filename starts with character
            image = cv.LoadImageM(f, cv.CV_LOAD_IMAGE_GRAYSCALE)
            self.add(char, image)


if __name__ == '__main__':
    import sys
    mt = MaskTable()
    mt.from_files(sys.argv[1:])
    mt.save()
