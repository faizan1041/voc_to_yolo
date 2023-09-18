import os
import shutil
import math

# Set your folder paths
source_folder = 'data'
train_folder = 'train'
val_folder = 'val'

if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(val_folder):
    os.makedirs(val_folder)

# List all files in the folder
all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
# Filter only the images (assuming jpg, JPG, and PNG are your image formats)
image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Shuffle the images to ensure a random split
import random
random.shuffle(image_files)

# Calculate the number of images to go into the validation set
num_val = math.ceil(0.10 * len(image_files))
val_images = image_files[:num_val]
train_images = image_files[num_val:]

# Copy images and their corresponding XML files
for img in train_images:
    # copy image
    shutil.copy2(os.path.join(source_folder, img), os.path.join(train_folder, img))
    # find and copy XML
    xml_file = os.path.splitext(img)[0] + '.xml'
    if xml_file in all_files:
        shutil.copy2(os.path.join(source_folder, xml_file), os.path.join(train_folder, xml_file))

for img in val_images:
    # copy image
    shutil.copy2(os.path.join(source_folder, img), os.path.join(val_folder, img))
    # find and copy XML
    xml_file = os.path.splitext(img)[0] + '.xml'
    if xml_file in all_files:
        shutil.copy2(os.path.join(source_folder, xml_file), os.path.join(val_folder, xml_file))
