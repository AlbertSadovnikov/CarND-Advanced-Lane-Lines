import cv2
import numpy as np
from lanelines.features import extract
from lanelines.curves import get_xy, quadratic_ransac_fit, lane_overlay, blend


def process(image, cal_data, pt_data, clf, state):
    undistorted = cv2.undistort(image, cal_data['matrix'], cal_data['distortion'])
    transformed = cv2.warpPerspective(undistorted, pt_data['matrix'],
                                      tuple(pt_data['target_size']), flags=cv2.INTER_LANCZOS4)

    features = extract(transformed)
    res = clf.predict(features.reshape(-1, features.shape[-1]))
    xy = get_xy(res.reshape(transformed.shape[:2]))
    try:
        coefficients_0, inliers_0 = quadratic_ransac_fit(xy[:, 0], xy[:, 1])
        coefficients_1, _ = quadratic_ransac_fit(xy[np.logical_not(inliers_0), 0],
                                                 xy[np.logical_not(inliers_0), 1])
        state.add_measurements(coefficients_0[0], coefficients_1[0])
    except ValueError:
        state.add_error(res)

    alpha, lanes = lane_overlay(state.left_lane, state.right_lane, transformed.shape[:2])
    inverse_transform = np.linalg.inv(pt_data['matrix'])
    alpha = cv2.warpPerspective(alpha, inverse_transform, (image.shape[1], image.shape[0]), flags=cv2.INTER_LANCZOS4)
    lanes = cv2.warpPerspective(lanes, inverse_transform, (image.shape[1], image.shape[0]), flags=cv2.INTER_LANCZOS4)
    result = blend(image, lanes, alpha, 0.25)
    return result
