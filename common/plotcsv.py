#!/usr/bin/env python

from collections import defaultdict

import matplotlib.pyplot as plt


def main(fp, indexes):
    columns = defaultdict(list)
    for line in fp:
        row = line.split(',')
        for n, t in enumerate(row):
            try:
                x = float(t)
            except ValueError:
                continue
            columns[n].append(x)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(columns[indexes[0]], columns[indexes[1]], 'x')
    plt.show()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1]) as fp:
        main(fp, [int(i) for i in sys.argv[2:]])
