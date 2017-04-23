import cv2
import numpy as np
from lanelines.features import extract
from lanelines.curves import get_xy, quadratic_ransac_fit, lane_overlay, blend


def process(image, cal_data, pt_data, clf):
    undistorted = cv2.undistort(image, cal_data['matrix'], cal_data['distortion'])
    transformed = cv2.warpPerspective(undistorted, pt_data['matrix'],
                                      tuple(pt_data['target_size']), flags=cv2.INTER_LANCZOS4)

    features = extract(transformed)
    res = clf.predict(features.reshape(-1, features.shape[-1]))
    res_prob = clf.predict_proba(features.reshape(-1, features.shape[-1]))
    binary = res_prob[:, 1].reshape(transformed.shape[:2])
    dsp = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
    dsp[:, :, 0] = 255*binary
    dsp[:, :, 1] = 255*binary
    dsp[:, :, 2] = 255*binary
    cv2.imshow('Binary', dsp)
    cv2.waitKey(1)
    xy = get_xy(res.reshape(transformed.shape[:2]))
    coefficients_0, inliers_0 = quadratic_ransac_fit(xy[:, 0], xy[:, 1])
    coefficients_1, _ = quadratic_ransac_fit(xy[np.logical_not(inliers_0), 0],
                                             xy[np.logical_not(inliers_0), 1])

    alpha, lanes = lane_overlay(coefficients_0, coefficients_1, transformed.shape[:2])
    inverse_transform = np.linalg.inv(pt_data['matrix'])
    alpha = cv2.warpPerspective(alpha, inverse_transform, (image.shape[1], image.shape[0]), flags=cv2.INTER_LANCZOS4)
    lanes = cv2.warpPerspective(lanes, inverse_transform, (image.shape[1], image.shape[0]), flags=cv2.INTER_LANCZOS4)
    result = blend(image, lanes, alpha, 0.25)
    return result
