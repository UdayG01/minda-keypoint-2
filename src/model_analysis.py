from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# ==============================
# CONFIGURATION
# ==============================
MODEL_PATH = "model/best.pt" 
VIDEO_PATH = "test_video/2025-10-24 13-09-37.mkv"
# If you know the specific Keypoint Indices from your training:
# Example: 0 for "Connector Tip", 1 for "Terminal Socket"
# Set to None if you want the script to auto-detect moving vs stationary objects
specific_kp_indices = None 

# ==============================
# INITIALIZATION
# ==============================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

# Store data for analysis
data_log = []

print("🚀 Starting Inference & Data Extraction...")

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model.predict(frame, verbose=False, conf=0.35)
    result = results[0]
    
    # Check if we have keypoints
    if result.keypoints is not None and result.keypoints.xy.numel() > 0:
        # Extract Keypoints (Move to CPU and Numpy)
        # Shape: (Num_Objects, Num_Keypoints, 2)
        kps = result.keypoints.xy.cpu().numpy()
        confs = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None

        # Logic: If your model detects 'Connector' and 'Terminal' as separate instances
        # We calculate the Centroid of every detected instance in this frame
        frame_data = {'frame': frame_idx}
        
        for i, obj_kps in enumerate(kps):
            # Calculate centroid of this object's keypoints
            # (Ignoring (0,0) points which indicate no detection)
            valid_points = obj_kps[~np.all(obj_kps == 0, axis=1)]
            
            if len(valid_points) > 0:
                cx, cy = np.mean(valid_points, axis=0)
                frame_data[f'obj_{i}_x'] = cx
                frame_data[f'obj_{i}_y'] = cy
                frame_data[f'obj_{i}_conf'] = np.mean(confs[i]) if confs is not None else 0
        
        data_log.append(frame_data)

    frame_idx += 1
    if frame_idx % 20 == 0:
        print(f"Processed {frame_idx} frames...")

cap.release()
print("✅ Data Extraction Complete. Generating Analysis...")

# ==============================
# DATA ANALYSIS
# ==============================
df = pd.DataFrame(data_log)

# Heuristic: Identify Stationary vs Moving Objects
# We calculate the variance of the X coordinate. Low variance = Terminal (Stationary).
object_indices = [col.split('_')[1] for col in df.columns if 'conf' in col]

obj_variances = {}
for idx in object_indices:
    if f'obj_{idx}_x' in df.columns:
        # Calculate variance of X position
        var = df[f'obj_{idx}_x'].var()
        obj_variances[idx] = var

# Sort objects: Lowest variance is likely the Terminal, Highest is the Connector
sorted_objs = sorted(obj_variances.items(), key=lambda item: item[1])

if len(sorted_objs) >= 2:
    static_idx = sorted_objs[0][0] # The Terminal
    moving_idx = sorted_objs[-1][0] # The Connector (Hand)
    
    print(f"Detected Static Object ID: {static_idx} (Variance: {sorted_objs[0][1]:.2f})")
    print(f"Detected Moving Object ID: {moving_idx} (Variance: {sorted_objs[-1][1]:.2f})")

    # Calculate Euclidean Distance between Moving and Static for every frame
    df['dx'] = df[f'obj_{moving_idx}_x'] - df[f'obj_{static_idx}_x']
    df['dy'] = df[f'obj_{moving_idx}_y'] - df[f'obj_{static_idx}_y']
    df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)
    
    # Calculate Velocity (Change in distance)
    df['velocity'] = df['distance'].diff().fillna(0)

    # ==============================
    # PLOTTING
    # ==============================
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Plot 1: Raw Positions (Visual verification)
    ax1.set_title("Raw X-Axis Movement (Identify the Approach)")
    ax1.plot(df['frame'], df[f'obj_{static_idx}_x'], label="Terminal (Static)", color='green', linestyle='--')
    ax1.plot(df['frame'], df[f'obj_{moving_idx}_x'], label="Connector (Moving)", color='blue')
    ax1.set_ylabel("X Pixel Coordinate")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Euclidean Distance (The Critical Metric)
    ax2.set_title("Distance between Connector and Terminal (The 'Dip' is the Push)")
    ax2.plot(df['frame'], df['distance'], color='red', linewidth=2)
    
    # --- VISUALIZING THRESHOLDS ---
    # Draw a hypothetical 'Contact Threshold' line
    # You will use this graph to adjust this value (e.g., 50 pixels)
    contact_threshold = df['distance'].min() + 20 
    ax2.axhline(y=contact_threshold, color='orange', linestyle='--', label=f'Potential Threshold (<{contact_threshold:.0f}px)')
    
    ax2.set_ylabel("Distance (pixels)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Stability/Velocity (To detect the "Stop")
    ax3.set_title("Velocity (Movement Speed) - Zero means 'Holding'")
    ax3.plot(df['frame'], df['velocity'], color='purple')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_ylabel("Pixel Change per Frame")
    ax3.set_xlabel("Frame Number")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    print("Analysis graphs generated. Use these to determine your 'OK' signal logic.")

else:
    print("❌ Not enough objects detected to perform pairwise analysis. Check model confidence or video quality.")