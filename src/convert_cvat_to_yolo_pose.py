import xml.etree.ElementTree as ET
import os

ANNOTATION_FILES = ["annotations1.xml", "annotations2.xml", "annotations3.xml"]  # list of CVAT XML files to convert
OUT_DIRS = ["keypoint_dataset/yolo_labels1", "keypoint_dataset/yolo_labels2", "keypoint_dataset/yolo_labels3"]

for OUT_DIR, ANNOTATION_FILE in zip(OUT_DIRS, ANNOTATION_FILES):
    os.makedirs(OUT_DIR, exist_ok=True)
    # class mapping (adjust if needed)
    label_dict = {
        "connector": 0,
        "terminal1": 1,
        "terminal2": 2,
        "terminal3": 3
    }


    tree = ET.parse(ANNOTATION_FILE)
    root = tree.getroot()

    for image in root.findall("image"):

        img_name = image.get("name")
        img_w = float(image.get("width"))
        img_h = float(image.get("height"))

        # ---- Extract all bounding boxes ----
        boxes = []
        for box in image.findall("box"):
            lbl = label_dict[box.get("label")]
            xtl, ytl = float(box.get("xtl")), float(box.get("ytl"))
            xbr, ybr = float(box.get("xbr")), float(box.get("ybr"))

            xc = ((xtl + xbr)/2) / img_w
            yc = ((ytl + ybr)/2) / img_h
            w  = (xbr - xtl) / img_w
            h  = (ybr - ytl) / img_h

            boxes.append({
                "class": lbl,
                "xc": xc, "yc": yc, "w": w, "h": h,
                "xtl": xtl/img_w, "ytl": ytl/img_h,
                "xbr": xbr/img_w, "ybr": ybr/img_h,
                "kps": []
            })

        # ---- Extract and assign keypoints ----
        for p in image.findall("points"):
            lbl = label_dict[p.get("label")]
            coords = p.get("points").split(";")        
            for c in coords:
                x,y = map(float,c.split(","))
                x /= img_w; y /= img_h

                # assign keypoint to closest box OF SAME CLASS
                best, best_dist = None, 99999
                for b in boxes:
                    if b["class"] != lbl: continue
                    bx, by = b["xc"], b["yc"]
                    dist = abs(bx-x) + abs(by-y)
                    if dist < best_dist:
                        best_dist, best = dist, b

                if best:
                    best["kps"].append([x,y,2])  # 2 = visible


        # ---- Write YOLO Pose TXT ----
        if boxes:
            with open(f"{OUT_DIR}/{img_name.replace('.jpg','.txt')}", "w") as f:
                for b in boxes:
                    line = f"{b['class']} {b['xc']} {b['yc']} {b['w']} {b['h']}"
                    for kp in b["kps"]: line += f" {kp[0]} {kp[1]} {kp[2]}"
                    f.write(line+"\n")

    print("\n✔ All annotations successfully converted to YOLO-Pose format")
    print(f"📁 Saved inside: {OUT_DIR}/")
