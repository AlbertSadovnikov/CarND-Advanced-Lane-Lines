import numpy as np
import cv2
import glob

np.set_printoptions(precision=4, suppress=True)
display = False

point_grid = (9, 6)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
object_points_ref = np.zeros((point_grid[0] * point_grid[1], 3), np.float32)
object_points_ref[:, :2] = np.mgrid[0:point_grid[0], 0:point_grid[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
object_points = []  # 3d points in real world space
image_points = []  # 2d points in image plane.

# subpixel chessboard corner detection criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# fix calibration images size
image_size = (1280, 720)

images = glob.glob('camera_cal/calibration*.jpg')

if display:
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)

# Step through the list and search for chessboard corners

for filename in images:
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, point_grid, None)

    # If found, add object points, image points
    if ret:
        object_points.append(object_points_ref)
        corners = cv2.cornerSubPix(gray, corners, (9, 9), (-1, -1), criteria)
        image_points.append(corners)

        # Draw and display the corners
        if display:
            img = cv2.drawChessboardCorners(img, point_grid, corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(0)

ret, mtx, dist, rot_vectors, trans_vectors = cv2.calibrateCamera(object_points, image_points, image_size, None, None)

np.savez('calibration', matrix=mtx, distortion=dist)

if display:
    cv2.destroyAllWindows()


