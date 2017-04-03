import cv2
import numpy as np


def get_line(img, text):
    cv2.namedWindow('lines', cv2.WINDOW_NORMAL)
    points = []
    cv2.putText(img, text, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255),  1, cv2.LINE_AA)

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.ellipse(img, (x, y), (3, 3), 0, 0, 360,  (0, 0, 255), 2)
            points.append((x, y))
            cv2.imshow('lines', img)

    cv2.setMouseCallback('lines', click)

    cv2.imshow('lines', img)

    cv2.waitKey(0)

    points = np.array(points)

    ml, cl = np.linalg.lstsq(np.vstack([points[:, 1], np.ones(len(points))]).T, points[:, 0])[0]

    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])
    return ml, cl, min_y, max_y


filename = 'test_images/straight_lines2.jpg'
# load image
image = cv2.imread(filename)
# load calibration file
data = np.load('calibration.npz')
# undistort
image = cv2.undistort(image, data['matrix'], data['distortion'])

sl, il, ml, xl = get_line(image.copy(), 'Put points on the left line')
sr, ir, mr, xr = get_line(image.copy(), 'Put points on the right line')

y0, y1 = min(ml, mr), max(xl, xr)
x00, x10 = sl*y0 + il, sl*y1 + il
x01, x11 = sr*y0 + ir, sr*y1 + ir
# cv2.line(image, (int(x00), y0), (int(x10), y1), (0, 255, 0), 2)
# cv2.line(image, (int(x01), y0), (int(x11), y1), (0, 255, 0), 2)


lane_width = 256
x_margin = 64
lane_length = 256
pts0 = np.array([[x00, y0], [x10, y1],
                 [x01, y0], [x11, y1]], np.float32)
pts1 = np.array([[x_margin, 0], [x_margin, lane_length],
                 [lane_width + x_margin, 0], [lane_width + x_margin, lane_length]], np.float32)
target_size = (lane_width + 2 * x_margin, lane_length)
M = cv2.getPerspectiveTransform(pts0, pts1)
dst = cv2.warpPerspective(image, M, target_size, flags=cv2.INTER_CUBIC)
cv2.namedWindow('birds_eye', cv2.WINDOW_NORMAL)
cv2.imshow('birds_eye', dst)

np.savez('perspective', matrix=M, target_size=target_size, pts0=pts0, pts1=pts1, source_size=image.shape[:2])

cv2.waitKey(0)

cv2.destroyAllWindows()

