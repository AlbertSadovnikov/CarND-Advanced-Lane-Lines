import numpy as np
import cv2


def color_mask(in_hls, h_range=(0, 255), l_range=(0, 255), s_range=(0, 255)):
    mask_h = np.zeros(in_hls.shape[:2], dtype=np.uint8)
    mask_s = np.zeros(in_hls.shape[:2], dtype=np.uint8)
    mask_v = np.zeros(in_hls.shape[:2], dtype=np.uint8)
    mask_h[np.where(np.logical_and(in_hls[:, :, 0] >= h_range[0], in_hls[:, :, 0] <= h_range[1]))] = 255
    mask_s[np.where(np.logical_and(in_hls[:, :, 1] >= s_range[0], in_hls[:, :, 1] <= s_range[1]))] = 255
    mask_v[np.where(np.logical_and(in_hls[:, :, 2] >= v_range[0], in_hls[:, :, 2] <= v_range[1]))] = 255
    and_mask = np.minimum(mask_h, np.minimum(mask_s, mask_v))

    # dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (5, 5))
    ret_mask = cv2.dilate(and_mask, kernel=kernel)
    return ret_mask


if __name__ == '__main__':
    img = cv2.imread('test_images/test5.jpg')
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS_FULL)

    #mask = color_mask(hls, h_range=hue_range, v_range=val_range)

    cv2.namedWindow('hue', cv2.WINDOW_NORMAL)
    cv2.namedWindow('lightness', cv2.WINDOW_NORMAL)
    cv2.namedWindow('saturation', cv2.WINDOW_NORMAL)
    cv2.imshow('hue', hls[:, :, 0])
    cv2.imshow('lightness', hls[:, :, 1])
    cv2.imshow('saturation', hls[:, :, 2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()







