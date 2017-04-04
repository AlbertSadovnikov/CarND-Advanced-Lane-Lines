import cv2


def lose_intensity(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # loose color accuracy
    hsv //= 16
    hsv *= 16
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

