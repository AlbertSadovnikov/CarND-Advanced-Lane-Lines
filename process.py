import argparse
import cv2
import numpy as np
# from color_processing import lose_intensity
import sys
import matplotlib.pyplot as plt
from plotting import imagesc

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input')
    ap.add_argument('--output')
    args = ap.parse_args()
    cap = cv2.VideoCapture(args.input)
    fps, w, h = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print('Reading %s (%d, %d)@%d' % (args.input, w, h, fps))

    # load calibration data
    cal_data = np.load('calibration.npz')
    # load perspective transform data
    pt_data = np.load('perspective.npz')

    # create output
    four_cc = cv2.VideoWriter_fourcc(*'x264')
    out = cv2.VideoWriter(args.output, 0x00000021,  fps, tuple(pt_data['target_size']))

    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Saturation', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Lightness', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Hue', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Laplacian', cv2.WINDOW_NORMAL)
    cv2.namedWindow('SobelX', cv2.WINDOW_NORMAL)
    cv2.namedWindow('SobelY', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Angle', cv2.WINDOW_NORMAL)
    hsv_hist = np.zeros((16, 16, 16), dtype=np.uint64)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # undistort
        undistorted = cv2.undistort(frame, cal_data['matrix'], cal_data['distortion'])

        # color
        hsl = cv2.cvtColor(undistorted, cv2.COLOR_BGR2HLS_FULL)

        # target plane
        tg = hsl[:, :, 1]

        laplacian = cv2.Laplacian(tg, cv2.CV_64F)

        krn = cv2.getDerivKernels(1, 0, 11)
        # res =
        #print(np.outer(sobel5x[0], sobel5x[1]).T)
        #sys.exit()
        sobelx = cv2.sepFilter2D(tg, cv2.CV_32F, krn[0], krn[1])

        sobely = cv2.sepFilter2D(tg, cv2.CV_32F, krn[1], krn[0])

        angle = np.arctan2(sobelx, sobely)

        # perspective transform
        dst = cv2.warpPerspective(undistorted, pt_data['matrix'], tuple(pt_data['target_size']), flags=cv2.INTER_CUBIC)

        # convert colors
        # dst = lose_intensity(dst)

        # hist = cv2.calcHist([dst], [0, 1, 2])

        # write data
        # out.write(dst)

        cv2.imshow('Original', frame)
        cv2.imshow('Hue', hsl[:, :, 0])
        cv2.imshow('Lightness', hsl[:, :, 1])
        cv2.imshow('Saturation', hsl[:, :, 2])
        imagesc('Laplacian', laplacian)
        imagesc('SobelX', np.abs(sobelx), colormap=cv2.COLORMAP_HSV)
        imagesc('SobelY', np.abs(sobely), colormap=cv2.COLORMAP_HSV)
        imagesc('Angle', np.abs(sobelx) + np.abs(sobely), colormap=cv2.COLORMAP_HSV)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

