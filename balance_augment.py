import os
import xml.etree.ElementTree as ET
import cv2
import albumentations as A

# Directories
ROOT_DIR = 'data/'
OUTPUT_DIR = 'BalancedOutput'
TRAIN_DIR = os.path.join(ROOT_DIR, 'train')
VAL_DIR = os.path.join(ROOT_DIR, 'val')
NEW_DIMS = (640, 640)

# List of acceptable image extensions
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']

# Augmentation function
def get_augmentation():
    return A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.ColorJitter(p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.RandomGamma(p=0.5),
        A.Blur(blur_limit=(3, 7), p=0.5),
        A.IAAAdditiveGaussianNoise(scale=(10, 50), p=0.5),
        A.HueSaturationValue(p=0.5),
        A.ChannelShuffle(p=0.5),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_id']))

# Parse XML function
def get_labels_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    labels = [obj.find('name').text for obj in root.findall('object')]
    return labels



# Modified function to generate new XML annotation files for augmented images
def create_xml_annotation(original_xml_path, filename, augmented_bboxes, class_names):
    tree = ET.parse(original_xml_path)
    root = tree.getroot()
    root.find("filename").text = filename
    for obj in root.findall("object"):
        root.remove(obj)
    
    for i, bbox in enumerate(augmented_bboxes):
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = class_names[i]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(bbox[0]))
        ET.SubElement(bndbox, "ymin").text = str(int(bbox[1]))
        ET.SubElement(bndbox, "xmax").text = str(int(bbox[2]))
        ET.SubElement(bndbox, "ymax").text = str(int(bbox[3]))
    
    return ET.tostring(root)

# Function to resize image and bounding boxes
def resize_image_and_bboxes(image, bboxes):
    h, w, _ = image.shape
    scale_x = NEW_DIMS[0] / w
    scale_y = NEW_DIMS[1] / h
    
    resized_image = cv2.resize(image, NEW_DIMS)
    resized_bboxes = []
    
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        resized_bboxes.append([xmin * scale_x, ymin * scale_y, xmax * scale_x, ymax * scale_y])
    
    return resized_image, resized_bboxes

# Main process for augmentation and writing images/XML
for folder in [TRAIN_DIR, VAL_DIR]:
    subfolder_name = os.path.basename(folder)
    output_subfolder = os.path.join(OUTPUT_DIR, subfolder_name)
    os.makedirs(output_subfolder, exist_ok=True)

    # Count class instances
    class_counts = {}
    for file in os.listdir(folder):
        if file.endswith('.xml'):
            xml_path = os.path.join(folder, file)

            try:
                labels = get_labels_from_xml(xml_path)
            except Exception as e:
                print(e)
                continue

            for label in labels:
                class_counts[label] = class_counts.get(label, 0) + 1

    max_count = max(class_counts.values())

    for file in os.listdir(folder):
        if file.lower().endswith(tuple(IMG_EXTENSIONS)):
            img_path = os.path.join(folder, file)
            xml_path = os.path.splitext(img_path)[0] + '.xml'

            try:
                labels = get_labels_from_xml(xml_path)
                image = cv2.imread(img_path)

                for label in labels:
                    bboxes = []
                    category_ids = []

                    parsed = []

                    try:
                        parsed = ET.parse(xml_path).getroot().findall('object')
                    except Exception as e:
                        print(e)
                        continue

                    for obj in parsed:
                        xmin = int(obj.find('bndbox').find('xmin').text)
                        ymin = int(obj.find('bndbox').find('ymin').text)
                        xmax = int(obj.find('bndbox').find('xmax').text)
                        ymax = int(obj.find('bndbox').find('ymax').text)
                        label_name = obj.find('name').text
                        bboxes.append([xmin, ymin, xmax, ymax])
                        category_ids.append(label_name)

                    diff = max_count - class_counts[label]
                    num_augmentations = int(((diff + class_counts[label]) / class_counts[label]))
                    # num_augmentations = int(class_counts[label] / (diff + 1))

                    print(f"Found {class_counts[label]} for {label}, generating {num_augmentations} augmentations!")

                    for i in range(num_augmentations):
                        augmented = get_augmentation()(image=image, bboxes=bboxes, category_id=category_ids)
                        aug_img, aug_bboxes = resize_image_and_bboxes(augmented['image'], augmented['bboxes'])
                        aug_name = f"{os.path.splitext(file)[0]}_aug{i}" + os.path.splitext(file)[1]
                        cv2.imwrite(os.path.join(output_subfolder, aug_name), aug_img)

                        # Save the augmented XML
                        new_xml_data = create_xml_annotation(xml_path, aug_name, aug_bboxes, category_ids)
                        new_xml_path = os.path.join(output_subfolder, os.path.splitext(aug_name)[0] + '.xml')
                        with open(new_xml_path, 'wb') as f:
                            f.write(new_xml_data)

                # Save the original image
                orig_img, orig_bboxes = resize_image_and_bboxes(image, bboxes)
                orig_name = file
                cv2.imwrite(os.path.join(output_subfolder, orig_name), orig_img)

                # Save the original XML
                new_xml_data = create_xml_annotation(xml_path, orig_name, orig_bboxes, category_ids)
                new_xml_path = os.path.join(output_subfolder, os.path.splitext(orig_name)[0] + '.xml')
                with open(new_xml_path, 'wb') as f:
                    f.write(new_xml_data)

            except Exception as e:
                print(e)
                continue

print("Processing complete!")
