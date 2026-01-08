import os, random, shutil, yaml

# ============================================================
# USER SETTINGS
# ============================================================
DATASET_DIR = "keypoint_dataset"

IMAGES_DIRS = [
    f"{DATASET_DIR}/frames1",
    f"{DATASET_DIR}/frames2",
    f"{DATASET_DIR}/frames3"
]

LABELS_DIRS = [
    f"{DATASET_DIR}/yolo_labels1",
    f"{DATASET_DIR}/yolo_labels2",
    f"{DATASET_DIR}/yolo_labels3"
]

OUTS = [
    f"{DATASET_DIR}/yolo_dataset1",
    f"{DATASET_DIR}/yolo_dataset2",
    f"{DATASET_DIR}/yolo_dataset3"
]

for IMAGES_DIR, LABELS_DIR, OUT in zip(IMAGES_DIRS, LABELS_DIRS, OUTS):
    print(f"\n🔄 Preparing YOLO dataset from: {IMAGES_DIR} and {LABELS_DIR}")
    VAL_SPLIT   = 0.05                         # 5% validation
    # ============================================================
    # CREATE OUTPUT DIRS
    # ============================================================
    os.makedirs(f"{OUT}/images/train", exist_ok=True)
    os.makedirs(f"{OUT}/images/val",   exist_ok=True)
    os.makedirs(f"{OUT}/labels/train", exist_ok=True)
    os.makedirs(f"{OUT}/labels/val",   exist_ok=True)

    # ------------------------------------------------------------
    # FILTER ONLY images that have YOLO annotations
    # ------------------------------------------------------------
    all_images = [f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg")]
    images = [img for img in all_images if os.path.exists(f"{LABELS_DIR}/{img.replace('.jpg','.txt')}")]

    skipped = len(all_images) - len(images)

    random.shuffle(images)
    val_count = int(len(images) * VAL_SPLIT)
    val_set = set(images[:val_count])

    print(f"\n📊 Total frames available: {len(all_images)}")
    print(f"📄 Annotations found for : {len(images)}")
    print(f"⚠ Skipped due to missing labels : {skipped}")
    print(f"📁 Train = {len(images)-val_count}   |   Val = {val_count}\n")

    # ============================================================
    # SPLIT INTO TRAIN / VAL
    # ============================================================
    for img in images:
        label = img.replace(".jpg",".txt")

        if img in val_set:
            shutil.copy(f"{IMAGES_DIR}/{img}",  f"{OUT}/images/val/{img}")
            shutil.copy(f"{LABELS_DIR}/{label}",f"{OUT}/labels/val/{label}")
        else:
            shutil.copy(f"{IMAGES_DIR}/{img}",  f"{OUT}/images/train/{img}")
            shutil.copy(f"{LABELS_DIR}/{label}",f"{OUT}/labels/train/{label}")


    # ============================================================
    # CREATE data.yaml
    # ============================================================
    data_yaml = {
        "train": "images/train",     # FIXED PATH
        "val":   "images/val",       # FIXED PATH
        "kpt_shape": [1,3],
        "names": {0:"connector", 1:"terminal1", 2:"terminal2", 3:"terminal3"}
    }


    with open(f"{OUT}/data.yaml","w") as f:
        yaml.dump(data_yaml,f)

    print("✔ YOLO dataset prepared successfully")
    print(f"📂 Output directory → {OUT}")
    print(f"📄 data.yaml saved at → {OUT}/data.yaml")

    print("\n🚀 TRAIN YOLO POSE MODEL:")
    print(f"yolo pose train model=yolo11n-pose.pt data={OUT}/data.yaml imgsz=640 epochs=100")
