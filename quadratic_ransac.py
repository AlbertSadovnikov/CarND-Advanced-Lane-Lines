import cv2
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import lanelines.features as features
import matplotlib.pyplot as plt
from lanelines.curves import lane_overlay, blend


"""
This script is used for training binarization classifier
"""

DISPLAY_TEST_IMAGES = True


def preprocess(img, cal, pt):
    undistorted = cv2.undistort(img, cal['matrix'], cal['distortion'])
    transformed = cv2.warpPerspective(undistorted, pt['matrix'], tuple(pt['target_size']), flags=cv2.INTER_LANCZOS4)
    return transformed


def quadratic_ransac_fit(x, y):
    poly_2 = PolynomialFeatures(degree=2)
    x_2 = poly_2.fit_transform(x.reshape(-1, 1))
    m = RANSACRegressor(LinearRegression(), loss="squared_loss", residual_threshold=150)
    m.fit(x_2, y.reshape(-1, 1))
    cc = m.estimator_.coef_[0]
    cc[0] = m.estimator_.intercept_
    return m.estimator_.coef_, m.inlier_mask_


def quadratic(x, coefficients):
    poly_2 = PolynomialFeatures(degree=2)
    x_2 = poly_2.fit_transform(x.reshape(-1, 1))
    return np.squeeze(x_2.dot(coefficients.T))


def get_xy(img):
    xy = np.vstack(np.nonzero(img)).T
    return xy

# def get_lane_lines(img):


if __name__ == '__main__':

    # load calibration data
    cal_data = np.load('data/calibration.npz')
    # load perspective transform data
    pt_data = np.load('data/perspective.npz')

    if DISPLAY_TEST_IMAGES:
        clf = joblib.load('data/binarizer.clf')
        test_images = ['test_images/straight_lines2.jpg',
                       'test_images/test2.jpg',
                       'test_images/test3.jpg',
                       'test_images/test4.jpg',
                       'test_images/test6.jpg',
                       'test_images/harder_challenge_0004.png',
                       'test_images/harder_challenge_0006.png',
                       'test_images/harder_challenge_0007.png',
                       'test_images/harder_challenge_0008.png',
                       'test_images/harder_challenge_0012.png']

        # cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        # cv2.namedWindow('Binary', cv2.WINDOW_NORMAL)
        for test_filename in test_images:
            original = cv2.imread(test_filename)
            image = preprocess(original, cal_data, pt_data)
            f = features.extract(image)
            res = clf.predict(f.reshape(-1, f.shape[-1]))
            xy = get_xy(res.reshape(image.shape[:2]))
            coefficients_0, inliers_0 = quadratic_ransac_fit(xy[:, 0], xy[:, 1])
            coefficients_1, _ = quadratic_ransac_fit(xy[np.logical_not(inliers_0), 0],
                                                     xy[np.logical_not(inliers_0), 1])

            alpha, lanes = lane_overlay(coefficients_0, coefficients_1, image.shape[:2])
            inv_tr = np.linalg.inv(pt_data['matrix'])
            alpha = cv2.warpPerspective(alpha, inv_tr, (original.shape[1], original.shape[0]), flags=cv2.INTER_LANCZOS4)
            lanes = cv2.warpPerspective(lanes, inv_tr, (original.shape[1], original.shape[0]), flags=cv2.INTER_LANCZOS4)
            img = blend(original, lanes, alpha, 0.25)
            plt.imshow(img[:, :, [2, 1, 0]])
            plt.show()

        # cv2.destroyAllWindows()
