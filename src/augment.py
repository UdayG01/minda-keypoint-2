import os
import cv2
import random
from tqdm import tqdm
import albumentations as A

# ==============================
# CONFIG
# ==============================
DATASET_ROOT = "merged_yolo_dataset"

IMG_DIR = os.path.join(DATASET_ROOT, "images/train")
LBL_DIR = os.path.join(DATASET_ROOT, "labels/train")

AUG_PER_IMAGE = 3
IMAGE_EXTS = [".jpg", ".png", ".jpeg"]

# ==============================
# ALBUMENTATIONS PIPELINE
# ==============================
transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.4),
        A.Rotate(limit=12, p=0.4),
        A.GaussNoise(var_limit=(5, 20), p=0.15),
        A.MotionBlur(blur_limit=3, p=0.15),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"]
    ),
    keypoint_params=A.KeypointParams(
        format="xy",
        remove_invisible=False
    )
)

# ==============================
# HELPERS
# ==============================
def find_image(stem):
    for ext in IMAGE_EXTS:
        p = os.path.join(IMG_DIR, stem + ext)
        if os.path.exists(p):
            return p
    return None


# ==============================
# AUGMENTATION LOOP
# ==============================
label_files = [f for f in os.listdir(LBL_DIR) if f.endswith(".txt") and "_aug_" not in f]
print(f"Found {len(label_files)} original images to augment")

for label_file in tqdm(label_files, desc="Augmenting"):
    stem = os.path.splitext(label_file)[0]

    img_path = find_image(stem)
    lbl_path = os.path.join(LBL_DIR, label_file)

    if img_path is None:
        continue

    image = cv2.imread(img_path)
    if image is None:
        continue
        
    h_img, w_img = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bboxes = []
    keypoints = []
    class_labels = []

    with open(lbl_path) as f:
        for line in f:
            parts = list(map(float, line.split()))
            if len(parts) != 8:
                continue

            cls = int(parts[0])
            cx, cy, w, h = parts[1:5]
            kpx, kpy, _ = parts[5:8]

            bboxes.append([cx, cy, w, h])
            keypoints.append((kpx * w_img, kpy * h_img))  # to pixels
            class_labels.append(cls)

    if not bboxes:
        continue

    for i in range(AUG_PER_IMAGE):
        try:
            aug = transform(
                image=image,
                bboxes=bboxes,
                keypoints=keypoints,
                class_labels=class_labels
            )

            if not aug["bboxes"]:
                continue

            aug_img = aug["image"]
            h2, w2 = aug_img.shape[:2]

            new_stem = f"{stem}_aug_{i}_{random.randint(1000,9999)}"
            out_img = os.path.join(IMG_DIR, new_stem + ".jpg")
            out_lbl = os.path.join(LBL_DIR, new_stem + ".txt")

            cv2.imwrite(out_img, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

            with open(out_lbl, "w") as f:
                for cls, box, kp in zip(
                    aug["class_labels"],
                    aug["bboxes"],
                    aug["keypoints"]
                ):
                    cx, cy, w, h = box
                    kpx, kpy = kp

                    # cx, cy, w, h are already normalized by Albumentations (format="yolo")
                    # but keypoints are in pixels (format="xy")
                    kpx_norm = kpx / w2
                    kpy_norm = kpy / h2

                    # clamp to [0, 1]
                    cx, cy = max(0, min(cx, 1)), max(0, min(cy, 1))
                    w, h   = max(0, min(w, 1)), max(0, min(h, 1))
                    kpx_norm, kpy_norm = max(0, min(kpx_norm, 1)), max(0, min(kpy_norm, 1))

                    f.write(
                        f"{int(cls)} "
                        f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f} "
                        f"{kpx_norm:.6f} {kpy_norm:.6f} 2\n"
                    )

        except Exception as e:
            # print(f"Error augmenting: {e}")
            continue

print("✅ Augmentation fixed and completed")
