import glob, shutil
import os
import pickle
import xml.etree.ElementTree as ET
from os import listdir, getcwd
from os.path import join


output_dir = ""
input_dirs = ['data/train', 'data/val']
classes = ['car', 'truck']


def write_data_yml(output_dir, classes):
    data_content = f"""train: {output_dir}/train/images
val: {output_dir}/val/images
# test: ../test/images

nc: {len(classes)}
names: {classes}
"""
    with open(os.path.join(output_dir, "data.yml"), "w") as f:
        f.write(data_content)


def getImagesInDir(dir_path):
    image_list = []
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.JPG','.PNG']
    
    for extension in allowed_extensions:
        image_list.extend(glob.glob(dir_path + f'/*{extension}'))

    return image_list


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(dir_path, output_path, image_path):
    basename = os.path.basename(image_path)
    basename_no_ext = os.path.splitext(basename)[0]


    in_file = open(dir_path + '/' + basename_no_ext + '.xml')
    out_file = open(f"{output_path}/{basename_no_ext}.txt", 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

cwd = getcwd()

for dir_path in input_dirs:
    full_dir_path = cwd + '/' + dir_path

    output_path_ = f"{output_dir}/{dir_path.split(os.path.sep)[-1]}"
    output_path_labels = f"{output_path_}/labels"
    output_path_images = f"{output_path_}/images"
    
    os.makedirs(output_path_labels, exist_ok=True)
    os.makedirs(output_path_images, exist_ok=True)


    image_paths = getImagesInDir(full_dir_path)
    list_file = open(output_path_ + '.txt', 'w')

    for image_path in image_paths:
        shutil.copyfile(image_path, f"{output_path_images}/{image_path.split(os.path.sep)[-1]}")

        list_file.write(f"{output_path_images}/{image_path.split(os.path.sep)[-1]}" + '\n')
        convert_annotation(full_dir_path, output_path_labels, image_path)
    list_file.close()

    write_data_yml(output_dir, classes)


    print("Finished processing: " + dir_path)
