#!/usr/bin/env python

import os
import string

import Image, ImageDraw, ImageFont


font_dirs = [
    '.',
    '/usr/share/fonts/truetype/freefont',
    '/usr/share/fonts/truetype/msttcorefonts'
    ]


def main(font_name, chars, image_size=128, font_size=96, outdir='img'):
    if font_name.endswith('.ttf'):
        font_name = font_name[:-4]
    font_path = find_font(font_name)
    font = ImageFont.truetype(font_path, font_size)
    image = Image.new('L', (image_size, image_size), "#000000")
    for c in chars:
        filename = os.path.join(outdir, '%s-%s.png' % (c, font_name))
        draw_and_save_image(font, c, image, filename)

def draw_and_save_image(font, char, image, filename):
    draw = ImageDraw.Draw(image)
    draw.rectangle([(0, 0), image.size], fill='#000000')
    x, y = font.getsize(char)
    x = (image.size[0] - x) / 2
    y = (image.size[1] - y) / 2
    draw.text((x, y), char, font=font, fill='#ffffff')
    image.save(filename, 'PNG')

def find_font(name):
    return '/usr/share/fonts/truetype/freefont/FreeSerif.ttf'


if __name__ == '__main__':
    import sys
    main('FreeSerif', string.ascii_letters)
