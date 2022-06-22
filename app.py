#!/usr/bin/env python3

import cv2
from utils import *
import argparse
from pathlib import Path
from datetime import datetime
from time import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', '--input-uri', metavar="URI", required=True, help=
    'URI to input stream\n'
    '1) image sequence (e.g. %%06d.jpg)\n'
    '2) video file (e.g. file.mp4)\n'
    '3) MIPI CSI camera (e.g. csi://0)\n'
    '4) USB camera (e.g. /dev/video0)\n'
    '5) RTSP stream (e.g. rtsp://<user>:<password>@<ip>:<port>/<path>)\n'
    '6) HTTP stream (e.g. http://<user>:<password>@<ip>:<port>/<path>)\n')
    required.add_argument('-c', '--config', metavar="FILE",
                          default=Path(__file__).parent / 'cfg' / 'tiny-yolov4.cfg',
                          help='path to config file (e.g. tiny-yolov4.cfg)')
    required.add_argument('-w', '--weights', metavar="FILE",
                          default=Path(__file__).parent / 'cfg' / 'yolov4-tiny.weights',
                          help='path to weights file (e.g. yolov4-tiny.weights)')
    required.add_argument('-o', '--output-uri', metavar="URI",
                          help='URI to output video file')
    required.add_argument('-s', '--show', default=True,
                          action='store_true', help='show visualizations')
    required.add_argument('-l', '--label', default=True,
                          action='store_true', help='perform pseudo-labeling of the detections.')
    required.add_argument('-lf', '--label-folder',  metavar="URI",
                          default=Path(__file__).parent / 'labels_folder',
                          help='URI to perform pseudo-labeling into, eg: labels_folder')
    required.add_argument('-sl', '--skip-label', type=int, default=5,
                          help='Number of frames to skip for pseudo-labeling.')
                          
    args = parser.parse_args()

    # capture the video input
    video = cv2.VideoCapture(args.input_uri)
    ret, frame = video.read()
    if not ret: # checking
        raise RuntimeError('Cannot read video stream!')

    # initialize the detector
    detector = Detector(weights=str(args.weights), config_file=str(args.config))

    frame_id = 0
    vid_out = []
    fps_avg = []
    while True:
        start_t = time()
        # read the image
        ret, frame = video.read()

        if not ret: # error or end of stream heck
           break

        # detect the image
        img_det, classes, scores, boxes = detector.detect(frame)

        # Perform pseudo-labeling every skip_label frames
        if args.label:
            if frame_id % args.skip_label == 0:
                write_labels(frame.copy(), classes,  boxes, frame_id,
                         args.label_folder, args.input_uri)

        # display the detections
        if args.show:
            cv2.imshow('Detection', cv2.resize(img_det,(1600,900)))

        # video writer
        if args.output_uri is not None:
            if frame_id==0:
                Height, Width = img_det.shape[:2]
                now = datetime.now()
                Path('./output').mkdir(parents=True, exist_ok=True)
                tmstmp = now.strftime("%Y-%m-%dT%H:%M")
                fps = video.get(cv2.CAP_PROP_FPS) if 15 < video.get(cv2.CAP_PROP_FPS)<=30 else 15
                vid_out = cv2.VideoWriter(str(Path(__file__).parent) + '/output/' + str(tmstmp) +
                                          str(args.output_uri) + '.avi',
                                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (Width, Height))
            vid_out.write(img_det)

        end_t = time() - start_t
        fps_avg.append((1/end_t))
        print(f"Frame id: {frame_id}, FPS: {np.mean(fps_avg):.2f}, Detections: {len(boxes)}", end="\r", flush=True)
        # refresh the fps average array for a much more accurate representation
        if frame_id%60==0:
            fps_avg = []
        frame_id+=1

        # If 'q' or 'Esc' keys are pressed, exit the program
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    video.release()
    print("")
    if args.output_uri is not None:
        print("Realising video output.")
        vid_out.release()
    if args.show:
        print("Camera/Video off.")
        cv2.destroyAllWindows()
    print("Program ended.")
    exit(0)

if __name__ == '__main__':
    main()
