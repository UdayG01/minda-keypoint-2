import json
import os
import shutil
import cv2
import random
import yaml
from pathlib import Path

# Set your paths and classes here
SOURCE_DIR = r"c:\Work\Renata\Minda KeyPoint\minda_keypoint_detection_yolov11\new_dataset\dataset"
OUTPUT_DIR = r"c:\Work\Renata\Minda KeyPoint\minda_keypoint_detection_yolov11\yolo_dataset"
CLASSES = ['c1', 't1', 't2', 't3', 'finger']
CLASS_MAP = {name: i for i, name in enumerate(CLASSES)}
TRAIN_RATIO = 0.9

# We're doing 1 keypoint per object. YOLO wants: <class> <box_params> <kpt_params>
# Since we only have points, we'll make a tiny square box around each one.
BOX_SIZE_RATIO = 0.02 # 2% of the image size feels about right for a point

def setup_directories():
    # Fresh start: blow away the old output and recreate the subfolders
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

def convert_dataset():
    # The main loop: shuffle the files, split them up, and process everything
    setup_directories()
    
    json_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.json')]
    random.shuffle(json_files)
    
    split_idx = int(len(json_files) * TRAIN_RATIO)
    train_files = json_files[:split_idx]
    val_files = json_files[split_idx:]
    
    print(f"Total files: {len(json_files)}")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    for batch_files, split in [(train_files, 'train'), (val_files, 'val')]:
        for json_file in batch_files:
            process_file(json_file, split)

    # Wrap up by creating the data.yaml file
    create_yaml()

def process_file(json_filename, split):
    json_path = os.path.join(SOURCE_DIR, json_filename)
    image_filename = json_filename.replace('.json', '.jpg')
    image_path = os.path.join(SOURCE_DIR, image_filename)
    
    if not os.path.exists(image_path):
        print(f"Warning: Image not found for {json_filename}, skipping.")
        return

    
    # Try to grab dims from JSON, if they are missing/zero, fall back to cv2
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    img_h = data.get('imageHeight')
    img_w = data.get('imageWidth')
    
    if not img_h or not img_w:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error reading image {image_path}")
            return
        img_h, img_w = img.shape[:2]

    # Grab each point and turn it into a YOLO row
    yolo_lines = []
    shapes = data.get('shapes', [])
    
    for shape in shapes:
        label = shape.get('label')
        if label not in CLASS_MAP:
            continue
            
        class_id = CLASS_MAP[label]
        points = shape.get('points', [])
        
        if not points:
            continue
            
        # Label gives us [[x, y]]
        px, py = points[0]
        
        # Convert to 0.0-1.0 range (Normalization)
        n_px = px / img_w
        n_py = py / img_h
        
        # Center the tiny box exactly on the point
        n_bw = BOX_SIZE_RATIO
        n_bh = BOX_SIZE_RATIO
        
        # Final row structure: class cx cy w h px py vis
        n_cx = n_px
        n_cy = n_py
        
        # Keep values inside 0-1
        n_cx = max(0, min(1, n_cx))
        n_cy = max(0, min(1, n_cy))
        
        vis = 2 # Visible
        
        line = f"{class_id} {n_cx:.6f} {n_cy:.6f} {n_bw:.6f} {n_bh:.6f} {n_px:.6f} {n_py:.6f} {vis}"
        yolo_lines.append(line)
        
    if yolo_lines:
        # Write out the .txt file
        label_filename = json_filename.replace('.json', '.txt')
        label_path = os.path.join(OUTPUT_DIR, 'labels', split, label_filename)
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
            
        # Bring the image over to the new folder
        dest_image_path = os.path.join(OUTPUT_DIR, 'images', split, image_filename)
        shutil.copy2(image_path, dest_image_path)

def create_yaml():
    # This generates the YAML config that YOLOv11 needs to find the data
    yaml_content = {
        'path': OUTPUT_DIR, # Use absolute path to avoid directory confusion
        'train': 'images/train',
        'val': 'images/val',
        'kpt_shape': [1, 3],
        'nc': len(CLASSES),
        'names': {i: name for i, name in enumerate(CLASSES)}
    }
    
    yaml_path = os.path.join(OUTPUT_DIR, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"Created {yaml_path}")

if __name__ == '__main__':
    convert_dataset()
