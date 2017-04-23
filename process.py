from moviepy.editor import VideoFileClip
from lanelines.pipeline import process
import numpy as np
from sklearn.externals import joblib
import cv2

# load calibration data
cal_data = np.load('data/calibration.npz')
# load perspective transform data
pt_data = np.load('data/perspective.npz')
# load binarizer
clf = joblib.load('data/binarizer.clf')

cv2.namedWindow('Binary', cv2.WINDOW_NORMAL)

def process_frame(frame):
    result = frame
    try:
        result = process(frame, cal_data, pt_data, clf)
    except ValueError as ve:
        print(ve)
        return frame
    return result


clip = VideoFileClip("./videos/harder_challenge_video.mp4")
output_video = "./result_video.mp4"
output_clip = clip.fl_image(process_frame)
output_clip.write_videofile(output_video, audio=False)
