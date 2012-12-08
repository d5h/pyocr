#!/usr/bin/env python

import random
import string
import sys

import cv

import Image

from common.bin import binary
from common.concom import connected_components
from common.genimg import draw_text, load_font


class DataGenerator(object):

    def __init__(self, fonts, chars, noise_images=[]):
        self.fonts = fonts
        self.chars = chars
        self.noise_images = noise_images

    def generate(self, output, num_samples=250):
        num_samples = self.generate_noise_points(output, num_samples)
        self.generate_data_points(output, num_samples)

    def generate_noise_points(self, output, num_samples):
        num_noise_points = 0
        for noise_img in self.noise_images:
            mat = cv.LoadImageM(noise_img, cv.CV_LOAD_IMAGE_GRAYSCALE)
            mat = binary(mat, invert=True)
            for com in connected_components(mat):
                if 10 < com.mask.rows and 5 < com.mask.cols:
                    num_noise_points += 1
                    output.write(",%s\n" % float_list_to_csv(params_from_component(com)))
                    if num_samples <= num_noise_points:
                        return num_samples

        return num_noise_points

    def generate_data_points(self, output, num_samples):
        image_size = 64
        font_size = 48
        image = Image.new('L', (image_size, image_size), "#000000")
        mat = cv.CreateMatHeader(image_size, image_size, cv.CV_8U)
        rot = cv.CreateMat(2, 3, cv.CV_32F)
        mapmat = cv.CreateMat(image_size, image_size, cv.CV_8U)
        points_per_font = num_samples / len(self.fonts)
        for f in self.fonts:
            font = load_font(f, font_size)
            for c in self.chars:
                for _ in range(points_per_font):
                    draw_text(font, c, image)
                    cv.SetData(mat, image.tostring())
                    cx = mat.cols / 2.0 + random.gauss(0, mat.cols / 30.0)
                    cy = mat.rows / 2.0 + random.gauss(0, mat.rows / 30.0)
                    angle = random.gauss(0, 1 / 15.0)
                    cv.GetRotationMatrix2D((cx, cy), angle, 1.0, rot)
                    cv.WarpAffine(mat, mapmat, rot)
                    coms = connected_components(mapmat)
                    com = largest_component(coms)
                    output.write(c + ",%s\n" % float_list_to_csv(params_from_component(com)))

def largest_component(coms):
    # FIXME: Not really the area, but it'll do.
    return sorted(coms, key=lambda c: c.mask.rows * c.mask.cols, reverse=True)[0]

def params_from_component(com, with_one=False):
    p = [1.0, com.intensity, com.x_sym, com.y_sym, com.xy_sym] + list(com.intensity_grid.reshape(-1))
    if with_one:
        return p
    return p[1:]

def float_list_to_csv(fs, fmt='%f'):
    return ','.join([fmt % f for f in fs])

def main(dg, output):
    if output:
        with open(output, 'w') as fp:
            dg.generate(fp, 50)
    else:
        dg.generate(sys.stdout)


def arg_split(s):
    return [a.strip() for a in s.split(',')]

if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('--output', '-o')
    ap.add_argument('--fonts', '-f', type=arg_split, default=['FreeSans'])
    ap.add_argument('--noise', '-n', type=arg_split, default=[])
    args = ap.parse_args()
    chars = string.ascii_letters + string.digits
    dg = DataGenerator(args.fonts, chars, args.noise)
    main(dg, args.output)
