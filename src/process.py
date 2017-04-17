import cv2
import numpy as np
from scipy.signal import deconvolve
from plotting import imagesc
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

"""

"""


def preprocess(img, cal, pt):
    undistorted = cv2.undistort(img, cal['matrix'], cal['distortion'])
    transformed = cv2.warpPerspective(undistorted, pt['matrix'], tuple(pt['target_size']), flags=cv2.INTER_CUBIC)
    return transformed


def extract_features(img):
    n_features = 2
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    feat = np.zeros((img.shape[0], img.shape[1], n_features))
    # feature 0, lightness, scaled to 0.0 to 1.0
    feat[:, :, 0] = hls[:, :, 1].astype(np.float) / 255
    # feature 1, saturation, scaled to 0.0 to 1.0
    feat[:, :, 1] = hls[:, :, 2].astype(np.float) / 255
    return feat


if __name__ == '__main__':
    # load calibration data
    cal_data = np.load('data/calibration.npz')
    # load perspective transform data
    pt_data = np.load('data/perspective.npz')

    images = [('test_images/test1.jpg', 'train_images/test1.png'),
              ('test_images/test5.jpg', 'train_images/test5.png'),
              ('test_images/straight_lines1.jpg', 'train_images/straight_lines1.png')]

    image_filename = images[0][0]
    mask_filename = images[0][1]
    # load image
    image = preprocess(cv2.imread(image_filename), cal_data, pt_data)
    # load mask
    mask = preprocess(cv2.cvtColor(cv2.imread(mask_filename), cv2.COLOR_BGR2GRAY), cal_data, pt_data)
    mask = cv2.normalize(mask.astype(np.float), None, 0.0, 1.0, cv2.NORM_MINMAX)

    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
    cv2.imshow('Original', image)
    cv2.imshow('Mask', mask)

    f = extract_features(image)
    for idx in range(f.shape[-1]):
        win_name = 'Feature%d' % idx
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        imagesc(win_name, f[:, :, idx])

    dt = DecisionTreeClassifier()
    x_data = f.reshape(-1, f.shape[-1])
    y_data = mask.flatten() > 0.5
    dt.fit(x_data, y_data)

    res = dt.predict_proba(f.reshape(-1, f.shape[-1]))
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    imagesc('Result', res[:, 1].reshape(mask.shape))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

