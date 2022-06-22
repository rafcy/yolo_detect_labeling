import os
import cv2
from pathlib import Path


def write_labels(img, classes, boxes, frame_num, output_folder, input_uri):
    # create output folder if not exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    # get the filename of the input video
    filename = Path(input_uri).stem
    imsize = img.shape[:2]
    # loop over the labels and save them in a txt file in YOLO format
    for i in range(len(boxes)):
        box = boxes[i]
        classid = classes[i]
        fileTXT = open(f"{output_folder}/{filename}_{frame_num}.txt", "a")  # append mode
        # create yolo label format
        x, y, w, h = box
        im_height, im_width = imsize
        height = float(format(h / im_height, '.5f'))  # ymax - ymin
        width = float(format(w / im_width, '.5f'))  # xmax - xmin
        xcenter = float(format(((x + int(w / 2)) / im_width), '.5f'))
        ycenter = float(format(((y + int(h / 2)) / im_height), '.5f'))
        # write to file (append) and close
        fileTXT.write(
            str(classid) + ' ' + str(xcenter) + ' ' + str(ycenter) + ' ' + str(width) + ' ' + str(height) + "\n")
        fileTXT.close()
    # save the clean image
    if img is not None:
        cv2.imwrite(f"{output_folder}/{filename}_{frame_num}.jpg", img)

