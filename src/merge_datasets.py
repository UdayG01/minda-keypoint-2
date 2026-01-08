import os
import shutil
import yaml  # Only needed to write YAML cleanly; built-in in most envs

# =====================================================
# CONFIG
# =====================================================
BASE_PATH = "keypoint_dataset"
DATASETS = ["yolo_dataset1", "yolo_dataset2", "yolo_dataset3"]
OUTPUT = "merged_yolo_dataset"

splits = ["train", "val"]

# =====================================================
# CREATE OUTPUT FOLDERS
# =====================================================
for split in splits:
    os.makedirs(f"{OUTPUT}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUTPUT}/labels/{split}", exist_ok=True)

# =====================================================
# MERGE LOOP
# =====================================================
for idx, ds in enumerate(DATASETS, 1):
    for split in splits:
        img_dir = f"{BASE_PATH}/{ds}/images/{split}"
        lbl_dir = f"{BASE_PATH}/{ds}/labels/{split}"

        if not os.path.exists(img_dir):
            print(f"[SKIPPED] No {split} folder in {ds}")
            continue

        for file in os.listdir(img_dir):
            base, ext = os.path.splitext(file)
            new_name = f"d{idx}_{base}{ext}"  # Make unique

            shutil.copy(f"{img_dir}/{file}", f"{OUTPUT}/images/{split}/{new_name}")

            label = base + ".txt"
            if os.path.exists(f"{lbl_dir}/{label}"):
                shutil.copy(f"{lbl_dir}/{label}", f"{OUTPUT}/labels/{split}/d{idx}_{label}")

print("\n📦 Merge Completed →", OUTPUT)

# =====================================================
# WRITE DATA.YAML
# =====================================================
yaml_content = {
        "train": "images/train",     # FIXED PATH
        "val":   "images/val",       # FIXED PATH
        "kpt_shape": [1,3],
        "names": {0:"connector", 1:"terminal1", 2:"terminal2", 3:"terminal3"}
    }


with open(f"{OUTPUT}/data.yaml", "w") as f:
    yaml.dump(yaml_content, f, default_flow_style=False)

print("📝 data.yaml created successfully!")
