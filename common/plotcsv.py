#!/usr/bin/env python

from collections import defaultdict

import matplotlib.pyplot as plt


def main(fp, i, j):
    blue = defaultdict(list)
    red = defaultdict(list)
    for line in fp:
        row = line.split(',')
        for n, t in enumerate(row):
            try:
                x = float(t)
            except ValueError:
                continue
            if line.startswith(','):
                red[n].append(x)
            else:
                blue[n].append(x)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Plot blue on top of red
    ax.plot(red[i], red[j], 'rx', blue[i], blue[j], 'bo')
    plt.show()


if __name__ == '__main__':
    from argparse import ArgumentParser
    import sys

    ap = ArgumentParser()
    ap.add_argument('--file', '-f')
    ap.add_argument('col1', type=int)
    ap.add_argument('col2', type=int)
    args = ap.parse_args()

    if args.file:
        with open(args.file) as fp:
            main(fp, args.col1, args.col2)
    else:
        main(sys.stdin, args.col1, args.col2)
