import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import cv2


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


def lane_overlay(lane0, lane1, shape):
    image = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    xx = np.arange(0, 256)
    yy_0 = quadratic(xx, lane0)
    yy_1 = quadratic(xx, lane1)
    points_0 = np.vstack((yy_0, xx)).T.astype(np.int32)
    points_1 = np.vstack((yy_1, xx)).T.astype(np.int32)
    cv2.fillPoly(image, [np.vstack((points_0, points_1[-1::-1, :]))], [0, 255, 0])
    cv2.polylines(image, [points_0, points_1], False, [0, 0, 255], thickness=10)
    alpha = np.zeros(image.shape[:2])
    alpha[np.nonzero(np.sum(image, axis=2))] = 1
    alpha = cv2.blur(alpha, (5, 5))
    return alpha, image


def blend(background, foreground, alpha, k=0.5):
    foreground = foreground.astype(float)
    background = background.astype(float)
    foreground = (k * alpha.T * foreground.T).T
    background = ((1.0 - k * alpha).T * background.T).T
    out = foreground + background
    return out.astype(np.uint8)

