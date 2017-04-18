import cv2
import numpy as np
from scipy.signal import deconvolve
from plotting import imagesc
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

"""

"""


def preprocess(img, cal, pt):
    undistorted = cv2.undistort(img, cal['matrix'], cal['distortion'])
    transformed = cv2.warpPerspective(undistorted, pt['matrix'], tuple(pt['target_size']), flags=cv2.INTER_CUBIC)
    return transformed


def extract_features(img):
    n_features = 8
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l, s = hls[:, :, 1], hls[:, :, 2]
    feat = np.zeros((img.shape[0], img.shape[1], n_features))
    # feature 0, lightness, scaled to 0.0 to 1.0
    idx = 0
    feat[:, :, idx] = l.astype(np.float) / 255
    # feature 1, saturation, scaled to 0.0 to 1.0
    idx += 1
    feat[:, :, idx] = s.astype(np.float) / 255
    # feature 2, y coordinate, scaled
    idx += 1
    for row_num in range(img.shape[0]):
        feat[row_num, :, idx] = row_num / img.shape[0]
    # feature 3, x coordinate, scaled
    idx += 1
    for col_num in range(img.shape[1]):
        feat[:, col_num, 3] = col_num / img.shape[1]
    # feature 3, Sobel x abs on lightness
    slx = cv2.Sobel(l, cv2.CV_32F, 1, 0, ksize=5) / 255
    idx += 1
    feat[:, :, idx] = slx
    # feature 4, Sobel y abs on lightness
    sly = cv2.Sobel(l, cv2.CV_32F, 0, 1, ksize=5) / 255
    idx += 1
    feat[:, :, idx] = sly
    # feature 5, Sobel x abs on saturation
    ssx = cv2.Sobel(s, cv2.CV_32F, 1, 0, ksize=5) / 255
    idx += 1
    feat[:, :, idx] = ssx
    # feature 6, Sobel y abs on saturation
    ssy = cv2.Sobel(s, cv2.CV_32F, 0, 1, ksize=5) / 255
    idx += 1
    feat[:, :, idx] = ssy
    return feat

if __name__ == '__main__':
    # load calibration data
    cal_data = np.load('data/calibration.npz')
    # load perspective transform data
    pt_data = np.load('data/perspective.npz')

    images = [('train_images/test1.jpg', 'train_images/test1_mask.png'),
              ('train_images/test5.jpg', 'train_images/test5_mask.png'),
              ('train_images/straight_lines1.jpg', 'train_images/straight_lines1_mask.png'),
              ('train_images/harder_challenge_0005.png', 'train_images/harder_challenge_0005_mask.png'),
              ('train_images/harder_challenge_0009.png', 'train_images/harder_challenge_0009_mask.png'),
              ('train_images/harder_challenge_0010.png', 'train_images/harder_challenge_0010_mask.png'),
              ('train_images/harder_challenge_0011.png', 'train_images/harder_challenge_0011_mask.png'),
              ('train_images/harder_challenge_0013.png', 'train_images/harder_challenge_0013_mask.png'),
              ('train_images/harder_challenge_0014.png', 'train_images/harder_challenge_0014_mask.png')]

    x_data = np.empty((0, 8), np.float)
    y_data = np.empty(0, np.int)

    # setup ml data
    for image_filename, mask_filename in images:
        # load image
        image = preprocess(cv2.imread(image_filename), cal_data, pt_data)
        # load mask
        mask = preprocess(cv2.cvtColor(cv2.imread(mask_filename), cv2.COLOR_BGR2GRAY), cal_data, pt_data)
        mask = cv2.normalize(mask.astype(np.float), None, 0.0, 1.0, cv2.NORM_MINMAX)

        # cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        # cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
        # cv2.imshow('Original', image)
        # cv2.imshow('Mask', mask)

        f = extract_features(image)
        # for idx in range(f.shape[-1]):
        #     win_name = 'Feature%d' % idx
        #     cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        #     imagesc(win_name, f[:, :, idx])
        # cv2.waitKey(0)
        # # dt = svm.LinearSVC(class_weight='balanced')
        x_data_sample = f.reshape(-1, f.shape[-1])
        y_data_sample = mask.flatten() > 0.5
        x_data = np.concatenate([x_data, x_data_sample])
        y_data = np.concatenate([y_data, y_data_sample])

    print(x_data.shape)
    print(y_data.shape)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0, stratify=y_data)

    dt = DecisionTreeClassifier(max_depth=32, class_weight="balanced")
    # dt = svm.LinearSVC(class_weight='balanced')
    dt.fit(x_train, y_train)
    print(dt.feature_importances_)

    y_predicted = dt.predict(x_test)
    ps = precision_score(y_test, y_predicted)
    rs = recall_score(y_test, y_predicted)
    print('Precision: %f, Recall: %f' % (ps, rs))

    test_images = ['test_images/straight_lines2.jpg',
                   'test_images/test2.jpg',
                   'test_images/test3.jpg',
                   'test_images/test4.jpg',
                   'test_images/test6.jpg',
                   'test_images/harder_challenge_0001.png',
                   'test_images/harder_challenge_0002.png',
                   'test_images/harder_challenge_0003.png',
                   'test_images/harder_challenge_0004.png',
                   'test_images/harder_challenge_0006.png',
                   'test_images/harder_challenge_0007.png',
                   'test_images/harder_challenge_0008.png',
                   'test_images/harder_challenge_0012.png']

    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Binary', cv2.WINDOW_NORMAL)
    for test_filename in test_images:
        image = preprocess(cv2.imread(test_filename), cal_data, pt_data)
        f = extract_features(image)
        res = dt.predict(f.reshape(-1, f.shape[-1]))
        cv2.imshow('Original', image)
        imagesc('Binary', res.reshape(image.shape[:2]))
        cv2.waitKey(0)

    cv2.destroyAllWindows()



