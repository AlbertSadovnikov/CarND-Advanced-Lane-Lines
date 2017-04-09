import numpy as np
import cv2


def color_mask(in_hsv, h_range=(0, 255), s_range=(0, 255), v_range=(0, 255)):
    mask_h = np.zeros(in_hsv.shape[:2], dtype=np.uint8)
    mask_s = np.zeros(in_hsv.shape[:2], dtype=np.uint8)
    mask_v = np.zeros(in_hsv.shape[:2], dtype=np.uint8)
    mask_h[np.where(np.logical_and(in_hsv[:, :, 0] >= h_range[0], in_hsv[:, :, 0] <= h_range[1]))] = 255
    mask_s[np.where(np.logical_and(in_hsv[:, :, 1] >= s_range[0], in_hsv[:, :, 1] <= s_range[1]))] = 255
    mask_v[np.where(np.logical_and(in_hsv[:, :, 2] >= v_range[0], in_hsv[:, :, 2] <= v_range[1]))] = 255
    and_mask = np.minimum(mask_h, np.minimum(mask_s, mask_v))

    # dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (5, 5))
    ret_mask = cv2.dilate(and_mask, kernel=kernel)
    return ret_mask

if __name__ == '__main__':
    img = cv2.imread('test_images/straight_lines1.jpg')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue_range = (0, 35)
    val_range = (128, 255)

    mask = color_mask(hsv, h_range=hue_range, v_range=val_range)

    cv2.imshow('sample', img)
    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()







