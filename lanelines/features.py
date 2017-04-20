import cv2
import numpy as np


def roll(img, roll_x, roll_y):
    rx = np.roll(img, roll_x, axis=1)
    ry = np.roll(rx, roll_y, axis=0)
    return ry


def color(img):
    n_features = 2
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l, s = hls[:, :, 1], hls[:, :, 2]
    feat = np.zeros((img.shape[0], img.shape[1], n_features))
    # lightness
    feat[:, :, 0] = l.astype(np.float32) / 255
    # saturation
    feat[:, :, 1] = s.astype(np.float32) / 255
    return feat


def sobel(img):
    n_features = 2
    feat = np.zeros((img.shape[0], img.shape[1], n_features))
    # sobel x direction, kernel size 5
    slx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    feat[:, :, 0] = slx
    # sobel y direction, kernel size 5
    sly = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    feat[:, :, 1] = sly
    return feat


def extract(img):
    feature_list = []
    # get color features
    cf = color(img)
    # append to feature list
    feature_list.append(cf)

    # compute sobel features
    for idx in range(cf.shape[2]):
        sample = cf[:, :, idx]
        sf = sobel(sample)
        feature_list.append(sf)

    return np.concatenate(feature_list, axis=2)


def shift(features, roll_list):
    feature_list = []
    for item in roll_list:
        feature_list.append(roll(features, *item))

    return np.concatenate(feature_list, axis=2)
