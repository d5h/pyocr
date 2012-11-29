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
    font = load_font(font_name, font_size)
    image = Image.new('L', (image_size, image_size), "#000000")
    for c in chars:
        filename = os.path.join(outdir, '%s-%s.png' % (c, font_name))
        draw_and_save_image(font, c, image, filename)

def load_font(font_name, font_size=96):
    if font_name.endswith('.ttf'):
        font_name = font_name[:-4]
    font_path = find_font(font_name)
    return ImageFont.truetype(font_path, font_size)

def draw_and_save_image(font, char, image, filename):
    draw_text(font, char, image)
    image.save(filename, 'PNG')

def draw_text(font, char, image):
    draw = ImageDraw.Draw(image)
    draw.rectangle([(0, 0), image.size], fill='#000000')
    x, y = font.getsize(char)
    x = (image.size[0] - x) / 2
    y = (image.size[1] - y) / 2
    draw.text((x, y), char, font=font, fill='#ffffff')

def find_font(name):
    for d in font_dirs:
        path = os.path.join(d, '%s.ttf' % name)
        if os.path.exists(path):
            return path
    raise ValueError("Can't find font: %s" % name)


if __name__ == '__main__':
    import sys
    main('FreeSansBold', string.ascii_letters + string.digits + ':-,.!')
