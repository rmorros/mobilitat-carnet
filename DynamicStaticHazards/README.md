# Artificial vision for kick-scooter

## Goal:

The project consists of a computer vision analysis of the conditions of the circulation of micromobility vehicles (bicycles and personal mobility vehicles or VMPs). This analysis is used to determine the conditions of the road, the presence of static or dynamic obstacles that could pose a danger to the drivers of these vehicles or to pedestrians.

## Pre-requisites
1) Python 3.6
2) Python3-Dev (For Ubuntu, `sudo apt-get install python3-dev`)
3) Numpy `pip3 install numpy`
4) Cython `pip3 install cython`
5) Optionally, OpenCV 3.x with Python bindings. (Tested on OpenCV 3.4.0)
    - You can use [this script](tools/install_opencv34.sh) to automate Open CV 3.4 installation (Tested on Ubuntu 16.04).
    - Performance of this approach is better than not using OpenCV.
    - Installations from PyPI distributions does not use OpenCV.
6) CUDA/10.0 availability and CUDNN/7.4
7) cmake >= 3.16

```
NOTE: Make sure CUDA_HOME environment variable is set.
```

## Installation:

In order to develop our project we have made use of some existing libraries:
    - [Yolov4 Daknet - Detector] (https://github.com/AlexeyAB/darknet) : Detection algorithm which was created in C language by DarkNet, but I have used the python wrapper due to project languages requirements. Used to detect objects in stipulated frames.
    - [Re3 - Tracker] (https://gitlab.com/danielgordon10/re3-tensorflow) : Tracker algorithm used for the objects detected by Yolov3 to fasten the algorithm in video processing.
    - [CentroidTracker - PyImageSearch] (https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/) : Tracker used to coordinate the identifications between the YOLO Detector and the Re3 Tracker. Lots of modifications have been made to the initial model.
    
Please do install some files in order to execute the code as expected:
    - YOLOv4 Weights and COCO configurations modules: download from [here](https://drive.google.com/file/d/1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT/view)
    - RE3 Configuration Files : install files from [here](http://bit.ly/2L5deYF), move it into Tracking/logs directory and extract its content.
    
Create virtual environment and use the requirements.txt for the correct versions installation :
```
pip install -r requirements.txt
```

## Usage:

There is plenty of arguments variations that can modificate the functioning of the video detector:
```
python full_system.py -i <path_inputvideo> [-o <output_video_name>] [--resultsFile <name_txt_file>] [--visualize BOOL] [--maxDisappeared MAXDISAPPEARED] [--maxDistance MAXDISTANCE] [--skipFrames SKIPFRAMES]
```

When executing the script for the first time, a Summary.txt file is created in out/ directory, where there will be automatically written a review of all executions made, with the configurations and the input video.
I have thought about an interesting way of organizing the videos and .txt configuration files detected through this ummary file.
Now we will explain how it really works:
  - Summary.txt → This file will have all contents of files detected and a brief resume of them. It will contain the indices of detections that have been done such as:
          Index    date    input_file    skipFrame     maxDisappeared     maxDistance
  - Directory with the index number and date that will be uniq and will contain the mp4 detected video and the txt file with the explanation:
      - .mp4 file with the --output name installed
      - .txt file with the --resultsFile name determined. It contains configurations, times, and also results.
