# Biliard Shot Analyzer
Analysis system for biliard shots.
## Features
Given an input video, the system can return:
- The video with a superimposed minimap on the bottom left corner.
- The video with tracked bounding boxes of the billiard balls.
- Segmentation mask of a video frame.
## Build
The source code is built using CMake.
## Run
The system is composed of three executables. To run each executable on the provided dataset run the following commands from the source code root:
- ```$ ./ build / generate videos ./ dataset /``` To generate the videos with superimposed minimap.
- ```$ ./ build / generate masks and detections ./ dataset /``` To generate segmentation masks and detections.
- ```$ ./ build / generate performance ./ dataset /``` To generate the mIoU and mAP performances.
