from re import L
import vsketch
import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

def longest(line, x, y):
    (x1, y1, x2, y2) = line
    d1 = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    d2 = math.sqrt((x2 - x) ** 2 + (y2 - y) ** 2)

    if (d1 > d2):
        return (x1, y1, x, y)
    else:
        return (x2, y2, x, y)

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
    check_for_overlaps = vsketch.Param(False)
    contour_levels = vsketch.Param(3)
    layer_separation = vsketch.Param(False)
    default_layer = vsketch.Param(0)

    raster = []

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("11x17in", landscape=False)
        #image = Image.open('/Users/moishe/frames/main-2/fast-2-00914-main.png')
        image = Image.open('/Users/moishe/Desktop/Source/other-mandala-1.jpg')
        # convert image to numpy array
        rgb_im = image.convert('RGB')
        self.data = np.array(rgb_im)

        data = np.asarray(rgb_im)
        print(data.shape)

        WIDTH = data.shape[0]
        HEIGHT = data.shape[1]

        orig_data = gaussian_filter(data, sigma=3)

        fig = plt.gcf()
        INCH_HEIGHT = 8
        INCH_WIDTH = INCH_HEIGHT * (WIDTH / HEIGHT)
        fig.set_size_inches(INCH_HEIGHT, INCH_WIDTH)
        plt.axis('off')
        vsk.scale(vsk.width / (data.shape[0] * 1.1))

        levels = []

        if self.layer_separation:
            layers = [0, 1, 2]
        else:
            layers = [self.default_layer]

        for layer in layers:
            contour_line_hash = {}
            data = np.rot90(orig_data[:, :, layer], 2)
            if not len(levels):
                levels = self.contour_levels
            x = plt.contour(range(0, HEIGHT), range(0, WIDTH), data, levels=levels,
                            alpha=0.8, linewidths=0.01, colors='black')
            levels = x.levels
            print(levels)

            contour_lines = []
            orth_lines = []

            for contour in x.allsegs:
                for line in contour:
                    for i in range(self.contour_step, len(line) - self.contour_step, self.contour_step):
                        if i + self.contour_step >= len(line):
                            continue

                        x = line[i][0]
                        y = line[i][1]
                        x1 = line[i - self.contour_step][0]
                        y1 = line[i - self.contour_step][1]
                        x2 = line[i + self.contour_step][0]
                        y2 = line[i + self.contour_step][1]

                        dx = -(x2 - x1)
                        dy = y2 - y1

                        length = math.sqrt(dx * dx + dy * dy)

                        if (length >= 1):
                            dy = dy * 1/length * self.line_length
                            dx = dx * 1/length * self.line_length

                            #vsk.line(x, y, (x - dy), (y - dx))
                            #vsk.line(x1, y1, x2, y2)
                            contour_lines.append((x, y, x - dy, y - dx))
                            if (x,y) in contour_line_hash:
                                print("hmm, collision")
                            contour_line_hash[(x,y)] = (x, y, x - dy, y - dx)
                            orth_lines.append((x1, y1, x2, y2))

            print("Processing %d lines" % len(orth_lines))

            if self.check_for_overlaps:
                for (i, line1) in enumerate(orth_lines):
                    if i % 1000 == 0:
                        print(i)
                    for (j, line2) in enumerate(contour_lines):
                        if i == j:
                            continue
                        (x1, y1, x2, y2) = line1
                        (x3, y3, x4, y4) = line2
                        intersections = edge_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
                        if intersections:
                            (new_x, new_y) = intersections
                            orth_lines[i] = (x1, y1, new_x, new_y) #longest(line1, new_x, new_y)
                            #orth_lines[j] = longest(line2, new_x, new_y)

            vsk.stroke(layer + 1)
            for line in orth_lines:
                (x, y, x1, y1) = line
                vsk.line(x, y, x1, y1)

            for line in contour_lines:
                (x, y, x1, y1) = line
                vsk.line(x, y, x1, y1)

            print(len(orth_lines))

    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    VskContourSketch.display()
