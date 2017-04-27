import numpy as np


class State:
    def __init__(self):
        self._measurements = list()
        self._error_images = list()
        self._left_lane_measurements = list()
        self._right_lane_measurements = list()
        self._left_lane = np.array([200, 0, 0])
        self._right_lane = np.array([400, 0, 0])

    def add_measurements(self, lane0, lane1):
        # order lanes by x0
        if lane0[0] < lane1[0]:
            left_candidate, right_candidate = lane0, lane1
        else:
            left_candidate, right_candidate = lane1, lane0

        # only candidates with certain x0
        if 50 < left_candidate[0] < 350 and np.abs(left_candidate[1]) < 1 and np.abs(left_candidate[2]) < 0.01:
            self._left_lane_measurements.append(left_candidate)
            if len(self._left_lane_measurements) < 5:
                self._left_lane = self._left_lane_measurements[-1]
            else:
                # some smoothing
                delta = self._left_lane_measurements[-1] - self._left_lane
                self._left_lane = self._left_lane + 0.2 * delta

        if 300 < right_candidate[0] < 526 and np.abs(right_candidate[1]) < 1 and np.abs(right_candidate[2]) < 0.01:
            self._right_lane_measurements.append(right_candidate)
            if len(self._right_lane_measurements) < 5:
                self._right_lane = self._right_lane_measurements[-1]
            else:
                # some smoothing
                delta = self._right_lane_measurements[-1] - self._right_lane
                self._right_lane = self._right_lane + 0.2 * delta

    @property
    def curvature(self):
        y = 128
        ym_per_pix = 20 / 256  # meters per pixel in y dimension

        left_curve = ((1 + (2 * self.left_lane[2] * y * ym_per_pix + self.left_lane[1]) ** 2) ** 1.5) \
                     / np.absolute(2 * self.left_lane[2])
        right_curve = ((1 + (2 * self.right_lane[2] * y * ym_per_pix + self.right_lane[1]) ** 2) ** 1.5) \
                      / np.absolute(2 * self.right_lane[2])
        # no need to report both, assuming lane lines are parallel
        return (left_curve + right_curve) / 2

    @property
    def displacement(self):
        center_x = (self.right_lane[0] + self.left_lane[0]) / 2
        xm_per_pix = 3.7 / 256  # meters per pixel in x dimension
        return (center_x - 576 / 2) * xm_per_pix

    def add_error(self, img):
        self._error_images.append(img)

    @property
    def right_lane(self):
        return self._right_lane

    @property
    def left_lane(self):
        return self._left_lane
