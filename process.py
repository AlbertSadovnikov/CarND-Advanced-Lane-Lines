from moviepy.editor import VideoFileClip
from lanelines.pipeline import process
import numpy as np
from sklearn.externals import joblib
from lanelines.state import State
from argparse import ArgumentParser
import cv2

# load calibration data
cal_data = np.load('data/calibration.npz')
# load perspective transform data
pt_data = np.load('data/perspective.npz')
# load binarizer
clf = joblib.load('data/binarizer.clf')
# state for temporal filtering
state = State()


def process_frame(frame):
    result = process(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), cal_data, pt_data, clf, state)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

parser = ArgumentParser(description='CarND Advanced lane finder')
parser.add_argument('--video', help='Path to video file', dest='video_filename', required=True)
parser.add_argument('--output', help='Path to output video file', dest='out_filename', required=True)
args = parser.parse_args()
clip = VideoFileClip(args.video_filename)
output_clip = clip.fl_image(process_frame)
output_clip.write_videofile(args.out_filename, audio=False, progress_bar=True)
