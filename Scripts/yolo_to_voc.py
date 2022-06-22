# Script to convert yolo annotations to voc format
import os
import xml.etree.cElementTree as ET
from PIL import Image
from shutil import copyfile
ANNOTATIONS_DIR_PREFIX = r"/annot_directory"

DESTINATION_DIR = "/annot_destination_directory"
DESTINATION_FOLDER = ""

CLASS_MAPPING = {
    '0': 'car',
    '1': 'bus',
    '2': 'truck',
    '3': 'motorbike'
    # Add your remaining classes here.
}
 

def create_root(file_prefix,image_format,path, width, height):
    root = ET.Element("annotations")
    ET.SubElement(root, "filename").text = "{}.{}".format(file_prefix,image_format)
    ET.SubElement(root, "path").text = os.path.join(path, "{}.{}".format(file_prefix,image_format))
    # ET.SubElement(root, "path").text = path + "{}.{}".format(file_prefix,image_format)
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    return root


def create_object_annotation(root, voc_labels):
    for voc_label in voc_labels:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = voc_label[0]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = str(0)
        ET.SubElement(obj, "difficult").text = str(0)
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(int(voc_label[1]))
        ET.SubElement(bbox, "ymin").text = str(int(voc_label[2]))
        ET.SubElement(bbox, "xmax").text = str(int(voc_label[3]))
        ET.SubElement(bbox, "ymax").text = str(int(voc_label[4]))
    return root


def create_file(file_prefix,image_format,path, width, height, voc_labels):
    root = create_root(file_prefix,image_format,path, width, height)
    root = create_object_annotation(root, voc_labels)
    tree = ET.ElementTree(root)
    test="{}/{}.xml".format(DESTINATION_FOLDER, file_prefix)
    tree.write("{}/{}.xml".format(DESTINATION_FOLDER, file_prefix))
    image_filename = "{}.{}".format(file_prefix,image_format)
    src,dst = "{}/{}".format(path,image_filename),"{}/{}".format(DESTINATION_FOLDER,image_filename)
    copyfile( src,dst) # copy the images


def read_file(path,file_path,image_format):
    file_prefix = file_path.split(".txt")[0] # get the filename only
    image_file_name = "{}.{}".format(file_prefix,image_format) # get the image filename
    img_path = os.path.join(path, image_file_name)
    img = Image.open(img_path)

    w, h = img.size
    prueba = "{}/{}".format(path, file_path)
    print(prueba)
    with open(prueba) as file:
        lines = file.readlines()
        voc_labels = []
        for line in lines:
            voc = []
            line = line.strip()
            data = line.split()
            voc.append(CLASS_MAPPING.get(data[0]))
            bbox_width = float(data[3]) * w
            bbox_height = float(data[4]) * h
            center_x = float(data[1]) * w
            center_y = float(data[2]) * h
            voc.append(center_x - (bbox_width / 2))
            voc.append(center_y - (bbox_height / 2))
            voc.append(center_x + (bbox_width / 2))
            voc.append(center_y + (bbox_height / 2))
            voc_labels.append(voc)
        create_file(file_prefix, image_format, path, w, h, voc_labels)
    print("Processing complete for file: {}".format(file_path))


def start():
    global DESTINATION_FOLDER
    train_test = "{}/{}".format(ANNOTATIONS_DIR_PREFIX,"train.txt"), "{}/{}".format(ANNOTATIONS_DIR_PREFIX,"test.txt")
    if not os.path.exists(DESTINATION_DIR):
        os.makedirs(DESTINATION_DIR)
    for folder in train_test:
        folder_type = os.path.basename(folder).split('.txt')[0]
        DESTINATION_FOLDER = "{}/{}".format(DESTINATION_DIR,folder_type)
        if not os.path.exists(DESTINATION_FOLDER):
            os.makedirs(DESTINATION_FOLDER)
        print("Converting annotations in file \"{}\" for folder \"{}\".".format(folder,folder_type))
        with open(folder, encoding='utf8') as f:
            for image_path in f: # reading images paths from the train file line by line
                image_format = image_path.split('.')[1].split('\n')[0]
                filepath_txt = image_path.split('.')[0] + '.txt'
                try:
                    if os.stat(filepath_txt).st_size > 0: # file not empty
                        print("Done")
                        filename = os.path.basename(filepath_txt)
                        path = os.path.dirname(filepath_txt)
                        read_file(path,filename,image_format)

                except:
                    print("Error while converting.")



if __name__ == "__main__":
    start()
