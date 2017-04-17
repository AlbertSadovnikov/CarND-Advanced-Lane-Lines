import cv2
import numpy as np
from scipy.signal import deconvolve
from plotting import imagesc
import matplotlib.pyplot as plt


"""
An attempt to deblur the image.
"""


def deblur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float)
    kernel = np.array([1, 2, 4, 2, 1])
    output = np.zeros((gray.shape[0], gray.shape[1] - len(kernel) + 1))
    for idx in range(gray.shape[0]):
        line = gray[idx, :]
        res, remainder = deconvolve(line, kernel)
        output[idx, :] = res
    plt.figure()
    plt.plot(gray[-1, :-1] - gray[-1, 1:])
    plt.figure()
    plt.plot(gray[0, :-1] - gray[0, 1:])
    plt.show()
    return output


if __name__ == '__main__':
    filename = 'test_images/straight_lines1.jpg'
    # load image
    image = cv2.imread(filename)
    # load calibration data
    cal_data = np.load('data/calibration.npz')
    # load perspective transform data
    pt_data = np.load('data/perspective.npz')
    # undistort
    undistorted = cv2.undistort(image, cal_data['matrix'], cal_data['distortion'])
    # transform
    transformed = cv2.warpPerspective(undistorted, pt_data['matrix'], tuple(pt_data['target_size']),
                                      flags=cv2.INTER_CUBIC)

    # deblur
    deblurred = deblur(transformed)

    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Undistorted', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Transformed', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Deblurred', cv2.WINDOW_NORMAL)

    cv2.imshow('Original', image)
    cv2.imshow('Undistorted', undistorted)
    cv2.imshow('Transformed', transformed)
    imagesc('Deblurred', deblurred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

