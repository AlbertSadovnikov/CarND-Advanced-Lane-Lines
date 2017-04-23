from lanelines.features import roll_image
from lanelines.plotting import imagesc
import cv2
import numpy as np


def preprocess(img, cal, pt):
    undistorted = cv2.undistort(img, cal['matrix'], cal['distortion'])
    transformed = cv2.warpPerspective(undistorted, pt['matrix'], tuple(pt['target_size']), flags=cv2.INTER_NEAREST)
    return transformed

# load calibration data
cal_data = np.load('data/calibration.npz')
# load perspective transform data
pt_data = np.load('data/perspective.npz')


img = cv2.cvtColor(preprocess(cv2.imread('test_images/test2.jpg'), cal_data, pt_data), cv2.COLOR_BGR2GRAY)
cv2.namedWindow('original', cv2.WINDOW_NORMAL)
cv2.imshow('original', img)

slx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5) / 255
cv2.namedWindow('rolled', cv2.WINDOW_NORMAL)
imagesc('rolled', roll_image(slx, -5, 0))
cv2.waitKey(0)
cv2.destroyAllWindows()





