import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import cv2


def curvature(a, b, x_s, y_s, y_m):
    #
    # Curvature formula from http://www.intmath.com/applications-differentiation/8-radius-curvature.php
    #
    # R(x) = ( (1 + f'(x)^2)^(3/2) ) / |f''(x)| ,
    # which in quadratic case:
    # R(x) = ( (1 + (2Ax + B) ^ 2) ^ (3/2) ) / |2A|
    #
    # y = Ax^2 + Bx + C
    #
    # y = k * y_m
    # x = l * x_m
    # substituting
    #
    # k * y_m = A(l*x_m)^2 + B * l * x_m + C
    # y_m = (A * l^2 / k) * x_m^2 + (B * l / k) * x_m + C / k
    #
    # A_m = A * l^2 / k
    # B_m = B * l / k
    # C_m = C / k
    #
    # to get curvature of the y_m(x_m) at x_point_m the formula will be
    #
    # R(x_m) = ( (1 + (2A_m * x_m + B_m) ^ 2) ^ (3/2) ) / |2A_m|
    #
    # substituting
    # R(x_m) = ( (1 + (2 * A * l^2 / k * x_m + B * l / k) ^ 2) ^ (3/2) ) / |2 * A * l^2 / k|
    #
    # in our case, x and y are swapped, l = y_s, k = x_s (since the parabola is x = f(y))
    #
    # so the formula is

    return (1 + (2 * a * (y_s ** 2) / x_s * y_m + b * y_s / x_s) ** 2) ** 1.5 / np.absolute(2 * a * y_s ** 2 / x_s)


def quadratic_ransac_fit(x, y):
    poly_2 = PolynomialFeatures(degree=2)
    x_2 = poly_2.fit_transform(x.reshape(-1, 1))
    m = RANSACRegressor(LinearRegression(), loss="squared_loss", residual_threshold=250)
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
    cv2.polylines(image, [points_0, points_1], False, [255, 0, 0], thickness=10)
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

