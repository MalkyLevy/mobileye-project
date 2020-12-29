import math


class Adapter:
    def __init__(self):
        self.grid = {}

    def filter_same_candidates(self, all_candidates):
        red_x, red_y, green_x, green_y = all_candidates[0], all_candidates[1], all_candidates[2], all_candidates[3]
        for r_x, r_y in zip(red_x, red_y):
            for g_x, g_y in zip(green_x, green_y):
                if r_x == g_x and r_y == g_y:
                    green_x -= g_x
                    green_y -= g_y

        return red_x, red_y, green_x, green_y

    def adapter_part_1_to_part_2(self, all_candidates):
        red_x, red_y, green_x, green_y = self.filter_same_candidates(all_candidates)
        candidates = []
        auxiliary = []
        for x, y in zip(red_x, red_y):
            candidates.append((x, y))
            auxiliary.append("R")
        for x, y in zip(green_x, green_y):
            candidates.append((x, y))
            auxiliary.append("G")

        return candidates, auxiliary

    def init_grid(self):
        self.grid = {}

    def add_candidate(self, x, y):
        # returns True if the point was accepted
        dist = 50000
        # compute the cell of the point
        ix = int(math.floor(x / dist))
        iy = int(math.floor(y / dist))

        # check cell and all neighbors
        for nhcell in ((ix - 1, iy - 1), (ix, iy - 1), (ix + 1, iy - 1),
                       (ix - 1, iy), (ix, iy), (ix + 1, iy),
                       (ix - 1, iy + 1), (ix, iy + 1), (ix + 1, iy + 1)):
            if nhcell in self.grid:
                for xx, yy in self.grid[nhcell]:
                    if (x - xx) ** 2 + (y - yy) ** 2 < dist:
                        # anoter existing point is too close
                        # return False
                        return yy, xx

        # the new point is fine
        # self.img_candidates.append((x, y))

        # we should also add it to the grid for future checks
        if (ix, iy) in self.grid:
            self.grid[(ix, iy)].append((x, y))
        else:
            self.grid[(ix, iy)] = [(x, y)]

        return True

    def adapter_part_2_to_part_3(self, cropped_imgs_predicts, candidates):
        tfl_candidates = {}

        self.init_grid()

        for i, predict in enumerate(cropped_imgs_predicts):
            if predict[1] > 0.8 and candidates[i][1] > 0 and candidates[i][0] > 0:
                if self.add_candidate(candidates[i][1], candidates[i][0]) is True:
                    tfl_candidates[candidates[i]] = 1

                else:
                    tfl_candidates[self.add_candidate(candidates[i][1], candidates[i][0])] += 1

        # sorted_candidates_by_visits_num = \
        #     sorted(tfl_candidates.keys(), key=lambda candidate: tfl_candidates[candidate], reverse=False)
        #
        # candidates_list = sorted_candidates_by_visits_num[:4]

        candidates_list = []

        for cand in tfl_candidates.keys():
            if tfl_candidates[cand] > 1:
                candidates_list.append(cand)

        return candidates_list
