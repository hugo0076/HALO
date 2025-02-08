import os
import shutil
from tqdm import tqdm

def reorganize_val_set(val_dir):
    # Read val_annotations.txt
    with open(os.path.join(val_dir, 'val_annotations.txt'), 'r') as f:
        val_annotations = f.readlines()

    # Create a dictionary to store filename to class mappings
    file_to_class = {}
    for line in val_annotations:
        parts = line.strip().split('\t')
        file_to_class[parts[0]] = parts[1]

    # Create directories for each class
    classes = set(file_to_class.values())
    for cls in classes:
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

    # Move images to their respective class directories
    images_dir = os.path.join(val_dir, 'images')
    for filename in tqdm(os.listdir(images_dir), desc="Moving files"):
        if filename in file_to_class:
            src = os.path.join(images_dir, filename)
            dst = os.path.join(val_dir, file_to_class[filename], filename)
            shutil.move(src, dst)

    # Remove the now-empty images directory
    os.rmdir(images_dir)
    # Remove the annotations file
    os.remove(os.path.join(val_dir, 'val_annotations.txt'))

# Usage
val_dir = 'data/tiny-imagenet-200/val'
reorganize_val_set(val_dir)