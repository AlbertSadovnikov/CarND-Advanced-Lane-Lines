## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


[corners]: ./output_images/corners_calibration2.jpg "Calibration pattern with corners detected"
[undistorted]: ./output_images/undistorted_calibration2.jpg "Undistorted calibration pattern"
[original_sample]: ./train_images/harder_challenge_0001.png "Sample image"
[undistorted_sample]: ./output_images/undistorted_harder_challenge_0001.png "Undistorted sample image"
[points_selection]: ./output_images/points_selection.png "Points selection"
[lane_area]: ./output_images/lane_area.png "Lane area"
[birds_eye]: ./output_images/birds_eye.png "Birds eye view on the lane area"
[sample_mask]: ./output_images/harder_challenge_0001_mask.png "Sample image mask"
[transformed_sample]: ./output_images/transformed45512.jpg "Transformed sample"
[thresholded_sample]: ./output_images/binary45512.jpg "Transformed sample"
[lines_detected]: ./output_images/lines55743.jpg "Detected lines"
[sample_result]: ./output_images/test12600.jpg "Sample result"

Goal of the project is to write a software pipeline to identify the lane boundaries in a video.


# The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform to create "birds-eye view" image.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

# Writeup
---

## Camera calibration
Camera calibration was done as it has been described in the instructions. Codes are available in
[calibrate.py](./calibrate.py).

Chessboard pattern with corners detected.
![corners]

Undistorted chessboard pattern.
![undistorted]

## Pipeline
### Distortion correction
Distortion correction is the first step of the processing pipeline (see [./lanelines/pipeline.py](./lanelines/pipeline.py)).
Here we apply derived distortion parameters.


Original sample.
![original_sample]

Undistorted sample.
![undistorted_sample]

### Transformation
Here I have actually made a change in pipeline and decided to do binarization after the perspective transformation.

For the transformation parameters estimation, I have implemented a small script to mark points on lane lines and
fit lines on those (see [./perspective.py](./perspective.py)).

Actual transformation can be found in the [./lanelines/pipeline.py](./lanelines/pipeline.py).

Points selection.
![points_selection]

Lane area.

![lane_area]

Birds eye view after perspective transformation.
![birds_eye]

### Binarization

For the thresholding (binarization) I have used the following features: saturation plane from HLS colorspace,
Sobel filter of the size 5 in the x and y directions over saturation and intensity planes.

Also, since edges do not quite match to color (if we want to detect color pixel near the edge), then there is a need
to add local information. So I have added 5px shifts in x and y directions (see extract method in [./lanelines/features.py](./lanelines/features.py)).

Since selecting a proper threshold could be quite tedious job - I've decided to use some supervised learning,
namely RandomForestClassifier. It is essentially a tree of thresholds.
For training it - I have manually marked lane lines on a few images.

Training code can be found in [./binarizer.py](./binarizer.py)

Sample training mask.
![sample_mask]

Transformed sample.
![transformed_sample]

Thresholded sample.
![thresholded_sample]


### Fitting polynomials

For fitting the quadratic polynomials I have used LinearRegression with RANSAC.
The idea is to fit one curve, then remove inliers and fit another curve.
The code can be found in [./lanelines/curves.py](./lanelines/curves.py).

Sample detected lines.
![lines_detected]


### Sample result
One of the few frames more or less correctly detected from the harder challenge.
![sample_result]

### Result videos
[./output_videos/result_project_video.mp4](./output_videos/result_project_video.mp4)
[./output_videos/result_challenge.mp4](./output_videos/result_challenge.mp4)
[./output_videos/result_harder_challenge_video.mp4](./output_videos/result_harder_challenge_video.mp4)

# Discussion
It was so far the hardest project in this term. Lane lines detection based on local (i.e. pixel values) seems to be hard.
There are a lot of problems with occlusions and scenes with a lot of light.

I think that employing CNNs here would give much better results, learning higher level concepts like lines, lane, etc. though training data collection can be quite time consuming.

Despite of spending around 3 weeks thinking and trying - I could not make it work robust enough.
