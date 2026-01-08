import os
import cv2
import random
import numpy as np

# ==============================
# CONFIG
# ==============================
DATASET_ROOT = "merged_yolo_dataset"
IMAGES_DIR = os.path.join(DATASET_ROOT, "images/train")
LABELS_DIR = os.path.join(DATASET_ROOT, "labels/train")
OUTPUT_DIR = "visualization_output"

CLASSES = ['connector', 'terminal1', 'terminal2', 'terminal3']
COLORS = [
    (255, 0, 0),   # Red
    (0, 255, 0),   # Green
    (0, 0, 255),   # Blue
    (255, 255, 0)  # Cyan
]

def draw_yolo(image, label_path):
    h, w = image.shape[:2]
    
    if not os.path.exists(label_path):
        return image

    with open(label_path, "r") as f:
        for line in f:
            parts = list(map(float, line.split()))
            if len(parts) < 8:
                continue
            
            cls_id = int(parts[0])
            cx, cy, bw, bh = parts[1:5]
            px, py, vis = parts[5:8]
            
            # Bbox to pixels
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            
            # Keypoint to pixels
            kpx = int(px * w)
            kpy = int(py * h)
            
            color = COLORS[cls_id % len(COLORS)]
            
            # Draw bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw class name
            label_text = f"{CLASSES[cls_id] if cls_id < len(CLASSES) else cls_id}"
            cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw keypoint
            cv2.circle(image, (kpx, kpy), 5, (255, 255, 255), -1) # White fill
            cv2.circle(image, (kpx, kpy), 2, color, -1)           # Color center
            
    return image

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"No images found in {IMAGES_DIR}")
        return

    # Select some original and some augmented images if possible
    orig_images = [f for f in image_files if "_aug_" not in f]
    aug_images = [f for f in image_files if "_aug_" in f]
    
    sample_size = 5
    to_visualize = random.sample(orig_images, min(len(orig_images), sample_size))
    to_visualize += random.sample(aug_images, min(len(aug_images), sample_size))
    
    print(f"Visualizing {len(to_visualize)} images to {OUTPUT_DIR}...")
    
    for img_name in to_visualize:
        img_path = os.path.join(IMAGES_DIR, img_name)
        lbl_name = os.path.splitext(img_name)[0] + ".txt"
        lbl_path = os.path.join(LABELS_DIR, lbl_name)
        
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        viz_image = draw_yolo(image, lbl_path)
        
        out_path = os.path.join(OUTPUT_DIR, f"viz_{img_name}")
        cv2.imwrite(out_path, viz_image)
        
    print(f"✅ Visualization done. Check the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    main()
