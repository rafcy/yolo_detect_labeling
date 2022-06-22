# yolo_detect_demo
Basic video object detection using OpenCV and a tiny yolo network.

## Packages Requirements
- Numpy 
- OpenCV

## Installation
 ```
 pip install numpy opencv-python
 ```

## Run the detector 
 ```
 python3 app.py -i ./video_file.mp4 -s 
   or
 python3 app.py -i ./video_file.mp4 -s -c ./cfg/tiny-yolov4.cfg -w ./cfg/yolov4-tiny.weights -o video_det 
 ```

## Run the pseudo-labeling and the Detector
 ```
 python3 app.py -i ./video_file.mp4 -s -l
   or
 python3 app.py -i ./video_file.mp4 -s -c ./cfg/tiny-yolov4.cfg -w ./cfg/yolov4-tiny.weights \ 
    -o video_det -l -lf labels_output -sl 10
 ```
 Options:
 - -i or --input-uri : input file or stream eg. "video_file.mp4" or "192.1.20.4/live"
 - -s or --show : display opencv window with detections/tracks
 - -w or --weights : yolo weights file path eg. ./cfg/tiny-yolov4.cfg
 - -c or --config: yolo configs file path eg. ./cfg/yolov4-tiny.weights
 - -o or --output-uri: video output filename, saves the file to ./output folder
 - -l or --label: perform pseudo-labeling on detections 
 - -lf or --label-folder: folder to create for saving the pseudo-labels (images and txt)
 - -sl or --skip-label: frames to skip when performing pseudo-labeling

 Class names can be found in './cfg/coco.names'
