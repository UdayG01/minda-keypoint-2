from ultralytics import YOLO
import cv2
import torch
import os

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "model/best.pt"            # trained model
VIDEO_PATH = "test_video/2025-10-24 13-09-37.mkv"
SAVE_PATH  = "outputs/2025-10-24 13-09-37.mp4"

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# ==============================
# LOAD MODEL + DEVICE
# ==============================
print(f"GPU Available: {torch.cuda.is_available()}")

device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH)                # loads model
model.to(device)                         # move to GPU if available

# ==============================
# VIDEO STREAM
# ==============================
cap = cv2.VideoCapture(VIDEO_PATH)
fps     = cap.get(cv2.CAP_PROP_FPS)
width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 🔥 Create video writer for SAVE_PATH
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(SAVE_PATH, fourcc, fps, (width, height))

print("\n🚀 Running Pose Inference... Press Q to stop early\n")

# ==============================
# FRAME BY FRAME INFERENCE
# ==============================
for result in model.predict(source=VIDEO_PATH, stream=True, device=device, imgsz=960, conf=0.35):

    frame = result.plot(labels=False, conf=False)        # draw keypoints & skeleton
    out.write(frame)             # <-- SAVE TO CUSTOM OUTPUT FILE
    cv2.imshow("YOLO Pose — Live Preview", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("\n🎥 Inference complete!")
print(f"Saved to:  {SAVE_PATH}")
print("YOLO default output also in: runs/pose/predict/")
