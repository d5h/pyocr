#!/usr/bin/env python

import cv


def connected_components(m):
    """m must be a binary image.  Returns a list of ConCom objects.
    Components only consist of ON pixels."""

    assert m.channels == 1

    label_mat = cv.CreateMat(m.rows, m.cols, cv.CV_32S)
    cv.Set(label_mat, 0)
    last_label = 0

    partitions = UnionFind()

    for r in range(m.rows):
        for c in range(m.cols):
            v = m[r, c]
            if v == 0:
                continue
            west = None
            north = None
            if 0 < c and m[r, c - 1]:  # == v:
                west = label_mat[r, c - 1]
                label_mat[r, c] = west
            if 0 < r and m[r - 1, c]:  # == v:
                north = label_mat[r - 1, c]
                label_mat[r, c] = north
            if west and north:
                partitions.union(west, north)
            if not (west or north):
                last_label += 1
                partitions.add(last_label, r, c)
                label_mat[r, c] = last_label
            p = partitions.get(west or north or last_label)
            p.xmin = min(c, p.xmin)
            p.ymin = min(r, p.ymin)
            p.xmax = max(c, p.xmax)
            p.ymax = max(r, p.ymax)

    components = []
    for p in partitions:
        com = ConCom.construct_from_partition(label_mat, p)
        components.append(com)

    return components

class UnionFind(object):

    def __init__(self):
        self.partitions = {}

    def union(self, a, b):
        if a != b:
            pa = self.partitions[a]
            pb = self.partitions[b]
            if pa is not pb:
                pa.xmin = min(pa.xmin, pb.xmin)
                pa.ymin = min(pa.ymin, pb.ymin)
                pa.xmax = max(pa.xmax, pb.xmax)
                pa.ymax = max(pa.ymax, pb.ymax)
                pa.labels.update(pb.labels)
                self.partitions[b] = pa
                for label in pb.labels:
                    self.partitions[label] = pa

    def add(self, label, r, c):
        p = Partition(label, r, c)
        self.partitions[label] = p

    def get(self, label):
        return self.partitions[label]

    def __iter__(self):
        s = set(self.partitions.values())
        return iter(s)

class Partition(object):

    def __init__(self, label, r, c):
        self.labels = {label}
        self.xmin = self.xmax = c
        self.ymin = self.ymax = r

class ConCom(object):

    @classmethod
    def construct_from_partition(cls, label_mat, part):
        w = part.xmax - part.xmin + 1
        h = part.ymax - part.ymin + 1
        com = cls(w, h)
        on_val = 255
        on = 0
        subl = label_mat[part.ymin: part.ymax + 1, part.xmin: part.xmax + 1]
        for r in range(subl.rows):
            for c in range(subl.cols):
                if subl[r, c] in part.labels:
                    com.mask[r, c] = on_val
                    on += 1
        com.offset = part.xmin, part.ymin
        com.intensity = float(on) / (w * h)

        com.x_sym = com.y_sym = com.xy_sym = 0.0
        for r in range(h):
            for c in range(w):
                if c < w / 2:
                    com.x_sym += xnor(com.mask[r, c], com.mask[r, w - c - 1])
                if r < h / 2:
                    com.y_sym += xnor(com.mask[r, c], com.mask[h - r - 1, c])
                    com.xy_sym += xnor(com.mask[r, c], com.mask[h - r - 1, w - c - 1])

        if w / 2 != 0:
            com.x_sym /= (w / 2) * h
        if h / 2 != 0:
            com.y_sym /= (h / 2) * w
            com.xy_sym /= (h / 2) * w

        return com

    def __init__(self, width, height):
        # Cache a copy of the mask with a border around it so we can
        # find contours.
        self.border_mask = cv.CreateMat(height + 2, width + 2, cv.CV_8U)
        self.mask = self.border_mask[1:-1, 1:-1]
        cv.Set(self.border_mask, 0)
        self.offset = (-1, -1)
        self.intensity = -1
        self.x_sym = -1
        self.y_sym = -1
        self.xy_sym = -1

def xnor(x, y):
    return bool(x) == bool(y)


if __name__ == '__main__':
    import sys
    from common.show import show as showimg
    i = cv.LoadImageM(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
    coms = connected_components(i)
    print len(coms)
    for c in sorted(coms, key=lambda c: c.mask.rows * c.mask.cols, reverse=True)[:15]:
        print c.intensity, c.x_sym, c.y_sym, c.xy_sym
        showimg(c.mask)
