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
        if 100 < left_candidate[0] < 300 and np.abs(left_candidate[1]) < 0.5 and np.abs(left_candidate[2]) < 0.001:
            self._left_lane_measurements.append(left_candidate)
            if len(self._left_lane_measurements) < 5:
                self._left_lane = self._left_lane_measurements[-1]
            else:
                # some smoothing
                delta = self._left_lane_measurements[-1] - self._left_lane
                self._left_lane = self._left_lane + 0.2 * delta

        if 320 < right_candidate[0] < 520 and np.abs(right_candidate[1]) < 0.5 and np.abs(right_candidate[2]) < 0.001:
            self._right_lane_measurements.append(right_candidate)
            if len(self._right_lane_measurements) < 5:
                self._right_lane = self._right_lane_measurements[-1]
            else:
                # some smoothing
                delta = self._right_lane_measurements[-1] - self._right_lane
                self._right_lane = self._right_lane + 0.2 * delta

    def add_error(self, img):
        self._error_images.append(img)

    @property
    def right_lane(self):
        return self._right_lane

    @property
    def left_lane(self):
        return self._left_lane
