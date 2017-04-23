import cv2
import numpy as np


def roll(img, roll_x, roll_y):
    rx = np.roll(img, roll_x, axis=1)
    ry = np.roll(rx, roll_y, axis=0)
    return ry


def sobel(img):
    feat = np.zeros((img.shape[0], img.shape[1], 2))
    # sobel x direction, kernel size 5
    slx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    feat[:, :, 0] = slx
    # sobel y direction, kernel size 5
    sly = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    feat[:, :, 1] = sly
    return feat


n_features = 37  # update this value, if you change the extract method


def extract(img):
    # convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    lightness, saturation = hls[:, :, 1], hls[:, :, 2]

    # compute sobel features for lightness
    sobel_lightness = sobel(lightness)

    # compute sobel for saturation channel
    sobel_saturation = sobel(saturation)

    # get shifted sobel features
    sobel_lightness_shifted = shift(sobel_lightness, [(0, 0), (0, 5), (5, 0), (0, -5), (-5, 0),
                                                      (-3, 3), (3, -3), (3, 3), (-3, -3)])
    sobel_saturation_shifted = shift(sobel_saturation, [(0, 0), (0, 5), (5, 0), (0, -5), (-5, 0),
                                                        (-3, 3), (3, -3), (3, 3), (-3, -3)])

    # combine features
    return np.concatenate((saturation[:, :, np.newaxis], sobel_lightness_shifted, sobel_saturation_shifted), axis=2)


def shift(features, roll_list):
    feature_list = []
    for item in roll_list:
        feature_list.append(roll(features, *item))

    return np.concatenate(feature_list, axis=2)
