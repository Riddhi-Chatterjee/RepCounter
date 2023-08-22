# RepCounter
A threshold independent, lightweight, repetition counting software built using OpenCV, MediaPipe and Scipy’s peak detector. The angle at some body joint is tracked, the data is filtered and peaks are detected which gives an idea of the repetition count. Further improvements are made in real time.

## Demo video:
https://drive.google.com/file/d/1-ImGsr3l-d3w9OKV4ROCFmh9RJcFcmPs/view?usp=sharing

## Local Version:

### Libraries required:
* math
* numpy
* sys
* time
* mediapipe
* OpenCV
* os
* scipy
* pandas
* matplotlib
* signal

### Steps to be followed:
Execute the command: "python3 RepCounter_v1.8.py", to execute the repetition counter script.

### Note:
* By default, the RepCounter takes real-time webcam feed as input. We can change this behaviour and perform repetition counting on video files by changing "cv2.VideoCapture(0)" to "cv2.VideoCapture(filename)" in the RepCounter_v1.8.py file.
* By default the RepCounter performs repetition counting for Push-Ups.
  * If we want to perform repetition counting for some other exercise then we need to change the angle being tracked. For example: The angle at the knee joint is tracked for Push-Ups, the angle at the shoulder needs to be tracked for exercises such as Shoulder-Abduction.
  * For changing the angle being tracked, change the contents of the "EssentialFeatures.csv" file. This file contains a descriptor of the angle being tracked. For example: The descriptor must be "2, A, 24, 26, 28, d" for Push-Ups and it must be "2, A, 14, 12, 24, d" for Shoulder Abduction.
  * Refer the FeatureExtraction-UserGuide and the MediapipePose image for understanding the syntax of feature descriptors.

## Web Version:

### Libraries required:
* scipy
* math
* numpy
* time
* sys

### Steps to be followed:
* It is recommended to install the Live Server extension of VS Code.
* Right click on the "RepCounter.html" file and click on "Open with Live Server". 

### Note:
By default the RepCounter performs repetition counting for Push-Ups.
  * If we want to perform repetition counting for some other exercise then we need to change the angle being tracked. For example: The angle at the knee joint is tracked for Push-Ups, the angle at the shoulder needs to be tracked for exercises such as Shoulder-Abduction.
  * For changing the angle being tracked, change the essential_features variable in the "Controller.py" file. This variable is a list containing a descriptor of the angle being tracked. For example: The descriptor must be "2, A, 23, 25, 27, d" for Push-Ups and it must be "2, A, 13, 11, 23, d" for Shoulder Abduction.
  * Refer the FeatureExtraction-UserGuide and the MediapipePose image for understanding the syntax of feature descriptors.
