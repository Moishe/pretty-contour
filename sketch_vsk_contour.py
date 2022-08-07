import bisect
from re import L
import vsketch
import math
import random

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

def longest(x1, y1, x2, y2, xi, yi):
    d1 = math.sqrt((x1 - xi) ** 2 + (y1 - yi) ** 2)
    d2 = math.sqrt((x2 - xi) ** 2 + (y2 - yi) ** 2)

    if d1 > d2:
        return (x1, y1, xi, yi)
    else:
        return (x2, y2, xi, yi)

def edge_intersection(x1: int, y1: int, x2: int, y2: int, x3: int, y3: int, x4: int, y4: int) -> list:
    """Intersection point of two line segments in 2 dimensions

    params:
    ----------
    x1, y1, x2, y2 -> coordinates of line a, p1 ->(x1, y1), p2 ->(x2, y2),

    x3, y3, x4, y4 -> coordinates of line b, p3 ->(x3, y3), p4 ->(x4, y4)

    Return:
    ----------
    list
        A list contains x and y coordinates of the intersection point,
        but return an empty list if no intersection point.

    """
    # None of lines' length could be 0.
    if ((x1 == x2 and y1 == y2) or (x3 == x4 and y3 == y4)):
        return []

    # The denominators for the equations for ua and ub are the same.
    den = ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))

    # Lines are parallel when denominator equals to 0,
    # No intersection point
    if den == 0:
        return []

    # Avoid the divide overflow
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / (den + 1e-16)
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / (den + 1e-16)

    # if ua and ub lie between 0 and 1.
    # Whichever one lies within that range then the corresponding line segment contains the intersection point.
    # If both lie within the range of 0 to 1 then the intersection point is within both line segments.
    if (ua < 0 or ua > 1 or ub < 0 or ub > 1):
        return []

    # Return a list with the x and y coordinates of the intersection
    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)
    return [x, y]


class VskContourSketch(vsketch.SketchClass):
    # Sketch parameters:
    # radius = vsketch.Param(2.0)

    contour_step = vsketch.Param(10)
    line_length = vsketch.Param(10)
    contour_levels = vsketch.Param(3)
    layer_separation = vsketch.Param(False)
    default_layer = vsketch.Param(0)
    draw_contour = vsketch.Param(True)
    draw_perpendicular = vsketch.Param(True)
    check_for_overlaps = vsketch.Param(False)
    invert_line_direction = vsketch.Param(False)

    raster = []
    lines = []
    points = []

    def insert_line(self, line):
        (t, i, x, y, x1, y1) = line
        line = (t, len(self.lines), x, y, x1, y1)
        self.lines.append(line)
        self.points.append((x, y, x <= x1, len(self.lines) - 1))
        self.points.append((x1, y1, x1 < x, len(self.lines) - 1))

    def truncate_lines(self):
        # build hash
        line_hash = defaultdict(list)
        for l in self.lines:
            (t, i, x, y, x1, y1) = l
            mix = math.floor(min(x, x1))
            miy = math.floor(min(y, y1))
            mxx = math.floor(max(x, x1) + 1)
            mxy = math.floor(max(y, y1) + 1)
            for xx in range(mix, mxx + 1):
                for yy in range(miy, mxy + 1):
                    line_hash[(xx, yy)].append(i)

        for (idx, l) in enumerate(self.lines):
            if idx % 1000 == 0:
                print(idx)
            (t, i, x, y, x1, y1) = l
            if t == 'c':
                continue
            mix = math.floor(min(x, x1))
            miy = math.floor(min(y, y1))
            mxx = math.floor(max(x, x1) + 1)
            mxy = math.floor(max(y, y1) + 1)
            for xx in range(mix, mxx + 1):
                for yy in range(miy, mxy + 1):
                    for lidx in line_hash[(xx, yy)]:
                        if lidx != i:
                            (t2, i2, x2, y2, x3, y3) = self.lines[lidx]
                            if t2 != 'c':
                                continue
                            intersection = edge_intersection(x, y, x1, y1, x2, y2, x3, y3)
                            if (intersection):
                                #(x, y, x1, y1) = (x, y, intersection[0], intersection[1])
                                (x, y, x1, y1) = longest(x, y, x1, y1, intersection[0], intersection[1])
                                new_line = (t, idx, x, y, x1, y1)
                                self.lines[idx] = new_line

    def build_lines(self, contours):
        for contour in contours:
            for line in contour:
                for i in range(self.contour_step, len(line) - self.contour_step, self.contour_step):
                    x = line[i][0]
                    y = line[i][1]
                    x1 = line[i - self.contour_step][0]
                    y1 = line[i - self.contour_step][1]
                    x2 = line[min(len(line), i + self.contour_step)][0]
                    y2 = line[min(len(line), i + self.contour_step)][1]

                    dx = -(x2 - x1)
                    dy = y2 - y1

                    length = math.sqrt(dx * dx + dy * dy)

                    if (length >= 1):
                        dy = dy * 1/length * self.line_length
                        dx = dx * 1/length * self.line_length

                        if self.invert_line_direction:
                            dy = -dy
                            dx = -dx

                        # put a little buffer in so everything doesn't intersect
                        x -= dy * 0.01
                        y -= dx * 0.01

                        self.insert_line(('p', i, x, y, x - dy, y - dx))
                        self.insert_line(('c', i, x1, y1, x2, y2))

    def draw_diminishing_line(self, vsk, x, y, x1, y1):
        dx = x1 - x
        dy = y1 - y
        vsk.line(x, y, x1, y1)


    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=True)
        image = Image.open('/Users/moishe/batch-output-14/sample-53-00001-main.png')
        #image = Image.open('/Users/moishe/Desktop/Source/other-mandala-bw.jpg')
        #image = Image.open('/Users/moishe/frames/a4-4b-00003-main.png')
        #image = Image.open('/Users/moishe/Desktop/Source/gradient-test.jpg')
        rgb_im = image.convert('RGB')
        self.data = np.array(rgb_im)

        data = np.asarray(rgb_im)
        print(data.shape)

        WIDTH = data.shape[0]
        HEIGHT = data.shape[1]

        orig_data = gaussian_filter(data, sigma=3)

        vsk.scale(min(vsk.width, vsk.height) / (data.shape[0] * 1.1))

        levels = []

        if self.layer_separation:
            layers = [0, 1, 2]
        else:
            layers = [self.default_layer]

        for layer in layers:
            self.lines = []

            data = np.rot90(orig_data[:, :, layer], 2)
            if not len(levels):
                levels = self.contour_levels
            x = plt.contour(range(0, HEIGHT), range(0, WIDTH), data, levels=levels,
                            alpha=0.8, linewidths=0.01, colors='black')
            levels = x.levels
            print(levels)

            self.build_lines(x.allsegs)

            if self.check_for_overlaps:
                print("Checking for overlaps in %d lines" % len(self.lines))
                self.truncate_lines()

        for line in self.lines:
            (t, i, x, y, x1, y1) = line
            if (t == 'p' and self.draw_perpendicular):
                vsk.stroke(2)
                self.draw_diminishing_line(vsk, x, y, x1, y1)
            if (t == 'c' and self.draw_contour):
                vsk.stroke(1)
                vsk.line(x, y, x1, y1)

    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")

if __name__ == "__main__":
    VskContourSketch.display()
