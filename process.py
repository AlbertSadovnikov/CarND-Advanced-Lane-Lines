from moviepy.editor import VideoFileClip
from lanelines.pipeline import process
import numpy as np
from sklearn.externals import joblib
from lanelines.state import State
import pickle

# load calibration data
cal_data = np.load('data/calibration.npz')
# load perspective transform data
pt_data = np.load('data/perspective.npz')
# load binarizer
clf = joblib.load('data/binarizer.clf')
# state for temporal filtering
state = State()


def process_frame(frame):
    result = process(frame, cal_data, pt_data, clf, state)
    return result


clip = VideoFileClip("./videos/project_video.mp4")
output_video = "./output_videos/project_video.mp4"
output_clip = clip.fl_image(process_frame)
output_clip.write_videofile(output_video, audio=False, progress_bar=True)
#with open('state_file.pkl', 'wb') as outfile:
#    pickle.dump(state, outfile, pickle.HIGHEST_PROTOCOL)
