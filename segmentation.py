import cv2
import sys
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
import numpy as np
import matplotlib.pyplot as plt


def model(input_size):
    m = Sequential()
    f = np.ones((3, 3, 3)) / 9
    w = np.zeros((3, 3, 3, 1))
    w[:, :, :, 0] = f
    print(w.shape)
    m.add(Conv2D(1, (3, 3), padding='same', input_shape=input_size, data_format='channels_last',
                 trainable=False,
                 weights=[w], use_bias=False))
    return m

filename = 'test_images/straight_lines1.jpg'


# img = cv2.imread(filename)

img = np.zeros((720, 1280, 3))
img[:, :, 0] = 10
img[:, :, 1] = 20
img[:, :, 2] = 30
md = model(img.shape)

res = md.predict(img[None, :, :, :])

print(np.squeeze(res[0, :5, :5, 0]))
#plt.imshow(np.squeeze(res))
#plt.show()

sys.exit()

im_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#im_hsv[:, :, 2] //= 2
#im_hsv[:, :, 2] += 64

im_res = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)

cv2.namedWindow('sample', cv2.WINDOW_NORMAL)

cv2.imshow('sample', im_res)

cv2.waitKey(0)

