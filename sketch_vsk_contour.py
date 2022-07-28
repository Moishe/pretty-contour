from re import L
import vsketch
import math
import random

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
    draw_flow_field = vsketch.Param(False)
    draw_contour = vsketch.Param(True)
    draw_perpendicular = vsketch.Param(True)

    raster = []

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=True)
        #image = Image.open('/Users/moishe/frames/main-2/fast-2-00914-main.png')
        #image = Image.open('/Users/moishe/batch-output-14/mains/sample-57-00001-main.png')
        #image = Image.open('/Users/moishe/batch-output-14/sample-26-00001-main.png')
        image = Image.open('/Users/moishe/frames/fast-4-00004-main.png')
        # convert image to numpy array
        rgb_im = image.convert('RGB')
        self.data = np.array(rgb_im)

        data = np.asarray(rgb_im)
        print(data.shape)

        WIDTH = data.shape[0]
        HEIGHT = data.shape[1]

        orig_data = gaussian_filter(data, sigma=3)

        #fig = plt.gcf()
        #INCH_HEIGHT = 8
        #INCH_WIDTH = INCH_HEIGHT * (WIDTH / HEIGHT)
        #fig.set_size_inches(INCH_HEIGHT, INCH_WIDTH)
        plt.axis('off')
        vsk.scale(vsk.width / (data.shape[0] * 1.4))

        levels = []

        if self.layer_separation:
            layers = [0, 1, 2]
        else:
            layers = [self.default_layer]

        for layer in layers:
            line_hash = defaultdict(list)
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

                            # put a little buffer in so everything doesn't intersect
                            x -= dy * 0.001
                            y -= dx * 0.001

                            #vsk.line(x, y, (x - dy), (y - dx))
                            #vsk.line(x1, y1, x2, y2)
                            orth_lines.append((x, y, x - dy, y - dx))
                            line_hash[int(x), int(y)].append((x, y, x - dy, y - dx))
                            line_hash[(int(x1),int(y1))].append((x1, y1, x2, y2))
                            contour_lines.append((x1, y1, x2, y2))

            if self.check_for_overlaps:
                print("Checking for overlaps with %d lines in hash containing %d points" % (len(orth_lines), len(line_hash)))
                for (i, line1) in enumerate(orth_lines):
                    if i % 1000 == 0:
                        print(i)
                    (x1, y1, x2, y2) = line1
                    for x in range(int(x1 - self.line_length * 2), int(x1 + self.line_length * 2 + 1)):
                        for y in range(int(y1 - self.line_length * 2), int(y1 + self.line_length * 2 + 1)):
                            if (x, y) in line_hash:
                                for line2 in line_hash[(x,y)]:
                                    (x3, y3, x4, y4) = line2

                                    intersections = edge_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
                                    if intersections:
                                        (new_x, new_y) = intersections
                                        orth_lines[i] = longest(line1, new_x, new_y) #(x2, y2, new_x, new_y) #

                print("Checking for overlaps with %d lines in hash containing %d points" % (len(contour_lines), len(line_hash)))
                for (i, line1) in enumerate(contour_lines):
                    if i % 1000 == 0:
                        print(i)
                    (x1, y1, x2, y2) = line1
                    for x in range(int(x1 - self.line_length * 2), int(x1 + self.line_length * 2 + 1)):
                        for y in range(int(y1 - self.line_length * 2), int(y1 + self.line_length * 2 + 1)):
                            if (x, y) in line_hash:
                                for line2 in line_hash[(x,y)]:
                                    (x3, y3, x4, y4) = line2

                                    intersections = edge_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
                                    if intersections:
                                        (new_x, new_y) = intersections
                                        contour_lines[i] = longest(line1, new_x, new_y) #(x2, y2, new_x, new_y) #

            #if self.draw_flow_field:
            #    self.draw_flow_field(vsk)

            if self.draw_perpendicular:
                vsk.stroke(1)
                for line in orth_lines:
                    (x, y, x1, y1) = line
                    vsk.line(x, y, x1, y1)

            if self.draw_contour:
                vsk.stroke(2)
                for line in contour_lines:
                    (x, y, x1, y1) = line
                    vsk.line(x, y, x1, y1)

            print(len(orth_lines))

    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")

    def draw_flow_field(self, vsk):
        FLOW_FIELD_WIDTH = int(HEIGHT / self.line_length)
        FLOW_FIELD_HEIGHT = int(WIDTH / self.line_length)
        flow_field = [(0, 0)] * (max(FLOW_FIELD_WIDTH, FLOW_FIELD_HEIGHT) ** 2)
        # seed it
        for line in orth_lines:
            (x, y, x1, y1) = line
            vector = (x1 - x, y1 - y)
            y = int(y / self.line_length)
            x = int(x / self.line_length)
            idx = y * FLOW_FIELD_WIDTH + x
            flow_field[idx] = vector

        RANGE = 2
        ACUITY = 1
        for i in range(0, 8):
            print(i)
            new_flow_field = flow_field.copy()
            for x in range(0, FLOW_FIELD_WIDTH):
                for y in range(0, FLOW_FIELD_HEIGHT):
                    idx = y * FLOW_FIELD_WIDTH + x
                    if (flow_field[idx] == (0,0)):
                        sum = [flow_field[idx][0] * ACUITY, flow_field[idx][1] * ACUITY]
                        count = ACUITY
                        for x1 in range(max(0, x - RANGE), min(FLOW_FIELD_WIDTH - 1, x + RANGE)):
                            for y1 in range(max(0, y - RANGE), min(FLOW_FIELD_HEIGHT - 1, y + RANGE)):
                                idx = y1 * FLOW_FIELD_WIDTH + x1
                                if (flow_field[idx] == (0,0)):
                                    continue
                                count += 1
                                sum[0] += flow_field[idx][0]
                                sum[1] += flow_field[idx][1]
                        new_flow_field[idx] = (sum[0] / count, sum[1] / count)
            flow_field = new_flow_field.copy()

        vsk.stroke(3)
        for x in range(0, FLOW_FIELD_WIDTH):
            for y in range(0, FLOW_FIELD_HEIGHT):
                idx = y * FLOW_FIELD_WIDTH + x
                xx = x * self.line_length
                yy = y * self.line_length
                #vsk.line(xx, yy, xx + flow_field[idx][0], yy + flow_field[idx][1])

        random.shuffle(orth_lines)
        for line in orth_lines:
            (x, y, dx, dy) = line
            dx = dx - x
            dy = dy - y
            #vsk.line(x, y, dx, dy)
            for i in range(0, 10):
                prev_x = x
                prev_y = y
                x += dx
                y += dy
                if (x < 0 or x > WIDTH or y < 0 or y > HEIGHT):
                    break

                ffx = int(x / self.line_length)
                ffy = int(y / self.line_length)
                idx = ffy * FLOW_FIELD_WIDTH + ffx

                if (flow_field[idx] != (0,0)):
                    dx = (dx + flow_field[idx][0]) / 2
                    dy = (dy + flow_field[idx][0]) / 2

                vsk.line(prev_x, prev_y, x, y)



if __name__ == "__main__":
    VskContourSketch.display()
