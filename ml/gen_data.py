#!/usr/bin/env python

import string
import sys

import cv

import Image

from common.concom import connected_components
from common.genimg import draw_text, load_font


class DataGenerator(object):

    def __init__(self, fonts, chars, noise_images=[]):
        self.fonts = fonts
        self.chars = chars
        self.noise_images = noise_images

    def generate(self, output):
        image_size = 64
        font_size = 48
        image = Image.new('L', (image_size, image_size), "#000000")
        mat = cv.CreateMatHeader(image_size, image_size, cv.CV_8U)
        for f in self.fonts:
            font = load_font(f, font_size)
            for c in self.chars:
                draw_text(font, c, image)
                cv.SetData(mat, image.tostring())
                coms = connected_components(mat)
                com = sorted(coms, key=lambda c: c.mask.rows * c.mask.cols, reverse=True)[0]
                output.write("%s,%f,%f,%f,%f\n" % (c, com.intensity, com.x_sym, com.y_sym, com.xy_sym))

        for noise_img in self.noise_images:
            mat = cv.LoadImageM(noise_img, cv.CV_LOAD_IMAGE_GRAYSCALE)
            for com in connected_components(mat):
                output.write(",%f,%f,%f,%f\n" % (com.intensity, com.x_sym, com.y_sym, com.xy_sym))

def main(fonts, chars, output):
    dg = DataGenerator(fonts, chars)
    if output:
        with open(output, 'w') as fp:
            dg.generate(fp)
    else:
        dg.generate(sys.stdout)


if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('--output', '-o')
    ap.add_argument('fonts', nargs='+')
    args = ap.parse_args()
    main(args.fonts, string.ascii_letters + string.digits, args.output)
