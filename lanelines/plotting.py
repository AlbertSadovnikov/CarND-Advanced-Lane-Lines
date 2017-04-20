import cv2
import numpy as np


def imagesc(window_name, im, colormap=cv2.COLORMAP_JET):
    display_img = cv2.applyColorMap(np.uint8(255.0 * (im - np.min(im)) / (np.max(im) - np.min(im))), colormap)
    cv2.imshow(window_name, display_img)

