import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import math
import json
import os

# ==============================
# CONFIGURATION
# ==============================
VIDEO_PATH = "test_video/2025-10-24 13-07-23.mkv"
MODEL_PATH = "model/best.pt"
OUTPUT_VIDEO_PATH = "analysis_outputs/push_detector2.mp4"
OUTPUT_IMAGE_PATH = "analysis_outputs/analysis_summary2.png"
ROI_CONFIG_FILE = "roi_config.json"

# Analysis Constants
BUFFER_SIZE = 100        # Frame history for graphs
PUSH_SPEED_THRESH = 10.0 # Speed to count as a "Push" (pixels/frame)
PROXIMITY_THRESH = 80.0  # Max distance between Connector & Terminal to validate a push
PUSH_COOLDOWN = 10       # Frames to wait before counting another push (Debounce)

# Layout Constants
TOTAL_WIDTH = 1280
TOTAL_HEIGHT = 720
PANEL_WIDTH = TOTAL_WIDTH // 2 

# ==============================
# SETUP
# ==============================
print(f"Loading model: {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

# --- ROI MANAGEMENT ---
rois = []
roi_states = []

def load_rois():
    if os.path.exists(ROI_CONFIG_FILE):
        with open(ROI_CONFIG_FILE, 'r') as f:
            return json.load(f)
    return []

def init_roi_states(roi_list):
    states = []
    for i, r in enumerate(roi_list):
        states.append({
            "data": r,
            "pushes_detected": 0,
            "status": "NG"
        })
    return states

# Initial Load
rois = load_rois()
roi_states = init_roi_states(rois)
print(f"Loaded {len(rois)} ROIs.")

# Output Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (TOTAL_WIDTH, TOTAL_HEIGHT))

# Buffers & State
speed_buffer = deque([0.0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
dist_buffer = deque([0.0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)

paused = False
last_push_frame = -999
frame_count = 0
last_dashboard_frame = None

# We need to keep track of the clean video frame for drawing
current_raw_frame = None 

print("=================================================")
print("CONTROLS:")
print("  'p' : Pause / Resume video")
print("  'r' : Draw ROIs (Define Terminals)")
print("  'q' : Quit")
print("=================================================")

while cap.isOpened():
    # Only read new frame if not paused
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break
        current_raw_frame = frame.copy()
        frame_count += 1
        
        # --- 1. INFERENCE ---
        results = model.track(frame, persist=True, verbose=False, conf=0.35, imgsz=960)
        
        # --- 2. LOGIC: KEYPOINT INTERACTION ---
        # Find Connector (Moving) & Terminals (Stationary)
        curr_positions = {}
        valid_ids = []
        max_speed = 0.0
        active_connector_id = None
        
        # Init prev_positions if needed
        if not hasattr(model, 'prev_positions'): model.prev_positions = {}

        if results[0].boxes.id is not None and results[0].keypoints is not None:
            ids = results[0].boxes.id.int().cpu().tolist()
            kps = results[0].keypoints.xy.cpu().numpy()
            valid_ids = ids

            for obj_id, kp_set in zip(ids, kps):
                valid_kps = [k for k in kp_set if k[0] > 0 and k[1] > 0]
                if valid_kps:
                    cx, cy = valid_kps[0]
                    curr_positions[obj_id] = (cx, cy)
                    
                    # Velocity Calc
                    speed = 0.0
                    if obj_id in model.prev_positions:
                        px, py = model.prev_positions[obj_id]
                        speed = math.hypot(cx - px, cy - py)
                    
                    if speed > max_speed:
                        max_speed = speed
                        active_connector_id = obj_id
                    
                    model.prev_positions[obj_id] = (cx, cy)
        
        # Cleanup
        model.prev_positions = {k: v for k, v in model.prev_positions.items() if k in valid_ids}

        # --- 3. DETECT PUSH ---
        current_nearest_dist = 300.0
        
        if active_connector_id is not None:
            c_pos = curr_positions[active_connector_id]
            
            # Find nearest other object (Terminal)
            nearest_term_pos = None
            min_dist = 9999.0
            
            for oid, pos in curr_positions.items():
                if oid != active_connector_id:
                    d = math.hypot(c_pos[0] - pos[0], c_pos[1] - pos[1])
                    if d < min_dist:
                        min_dist = d
                        nearest_term_pos = pos
            
            current_nearest_dist = min(min_dist, 300.0)

            # PUSH LOGIC
            if max_speed > PUSH_SPEED_THRESH:
                if (frame_count - last_push_frame) > PUSH_COOLDOWN:
                    if nearest_term_pos and min_dist < PROXIMITY_THRESH:
                        # PUSH DETECTED
                        last_push_frame = frame_count
                        
                        # UPDATE STATUS based on which ROI contains the terminal
                        tx, ty = nearest_term_pos
                        for r_state in roi_states:
                            rx, ry, rw, rh = r_state["data"]["rect"]
                            if rx <= tx <= rx + rw and ry <= ty <= ry + rh:
                                r_state["pushes_detected"] += 1
                                if r_state["pushes_detected"] >= 2:
                                    r_state["status"] = "OK"
                                print(f"Push on {r_state['data']['label']} (Total: {r_state['pushes_detected']})")
                                break # Count for only one ROI

        # Update Graphs
        speed_buffer.append(max_speed)
        dist_buffer.append(current_nearest_dist)
        
        # Store for display during pause
        last_annotated_frame = frame.copy()
    else:
        # If paused, we just reuse the last processed frame for display
        if current_raw_frame is not None:
            # Add a slight dim effect to indicate pause
            last_annotated_frame = cv2.addWeighted(current_raw_frame, 0.7, np.zeros_like(current_raw_frame), 0.3, 0)
            cv2.putText(last_annotated_frame, "PAUSED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 200), 5)
            cv2.putText(last_annotated_frame, "Press 'r' to Define Terminals", (55, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            last_annotated_frame = np.zeros((TOTAL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)

    # --- 4. VISUALIZATION: VIDEO PANEL (LEFT) ---
    video_view = last_annotated_frame.copy()
    
    # Draw ROIs
    for r_state in roi_states:
        rx, ry, rw, rh = r_state["data"]["rect"]
        color = (0, 0, 255) # Red (NG)
        if r_state["status"] == "OK":
            color = (0, 255, 0) # Green
        elif r_state["pushes_detected"] == 1:
            color = (0, 255, 255) # Yellow
            
        cv2.rectangle(video_view, (rx, ry), (rx+rw, ry+rh), color, 2)
        cv2.putText(video_view, r_state["data"]["label"], (rx, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    video_panel = cv2.resize(video_view, (PANEL_WIDTH, TOTAL_HEIGHT))
    
    # Overlay Dashboard
    if roi_states:
        dash_w, dash_h = 240, 50 + (len(roi_states) * 35)
        dx, dy = PANEL_WIDTH - dash_w - 20, 20
        
        overlay = video_panel.copy()
        cv2.rectangle(overlay, (dx, dy), (dx+dash_w, dy+dash_h), (10, 10, 10), -1)
        video_panel = cv2.addWeighted(overlay, 0.7, video_panel, 0.3, 0)
        
        cv2.putText(video_panel, "TERMINAL STATUS", (dx+10, dy+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        for i, r_state in enumerate(roi_states):
            y = dy + 70 + (i * 35)
            label = r_state["data"]["label"]
            status = r_state["status"]
            pushes = r_state["pushes_detected"]
            
            col = (100, 100, 255) if status == "NG" else (100, 255, 100)
            cv2.putText(video_panel, f"{label}: {status}", (dx+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            
            # Progress Dots
            for p in range(2):
                dot_col = col if p < pushes else (50, 50, 50)
                cv2.circle(video_panel, (dx + 200 + (p*15), y-5), 5, dot_col, -1)

    # --- 5. VISUALIZATION: GRAPHS (RIGHT) ---
    dashboard = np.full((TOTAL_HEIGHT, PANEL_WIDTH, 3), (20, 20, 20), dtype=np.uint8)
    margin = 40
    gh = (TOTAL_HEIGHT - 3*margin) // 2
    
    # Top Graph: Velocity
    g1t, g1b = margin, margin + gh
    cv2.rectangle(dashboard, (margin, g1t), (PANEL_WIDTH-margin, g1b), (35, 35, 35), -1)
    cv2.putText(dashboard, "VELOCITY (Px/Frame)", (margin, g1t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
    
    if len(speed_buffer) > 1:
        pts = []
        for i, val in enumerate(speed_buffer):
            x = margin + int(i*(PANEL_WIDTH-2*margin)/BUFFER_SIZE)
            y = g1b - int(min(val, 50.0)/50.0 * gh)
            pts.append((x,y))
        cv2.polylines(dashboard, [np.array(pts)], False, (0,255,255), 2, cv2.LINE_AA)
        
        th_y = g1b - int((PUSH_SPEED_THRESH/50.0)*gh)
        cv2.line(dashboard, (margin, th_y), (PANEL_WIDTH-margin, th_y), (0,0,150), 1)

    # Bottom Graph: Proximity
    g2t, g2b = g1b + margin, g1b + margin + gh
    cv2.rectangle(dashboard, (margin, g2t), (PANEL_WIDTH-margin, g2b), (35, 35, 35), -1)
    cv2.putText(dashboard, "PROXIMITY (Dist to Terminal)", (margin, g2t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
    
    if len(dist_buffer) > 1:
        pts = []
        for i, val in enumerate(dist_buffer):
            x = margin + int(i*(PANEL_WIDTH-2*margin)/BUFFER_SIZE)
            y = g2b - int(min(val, 300.0)/300.0 * gh)
            pts.append((x,y))
        cv2.polylines(dashboard, [np.array(pts)], False, (0,255,0), 2, cv2.LINE_AA)
        
        th_y2 = g2b - int((PROXIMITY_THRESH/300.0)*gh)
        cv2.line(dashboard, (margin, th_y2), (PANEL_WIDTH-margin, th_y2), (0,150,0), 1)

    # Combine & Show
    last_dashboard_frame = np.hstack((video_panel, dashboard))
    out.write(last_dashboard_frame)
    cv2.imshow("Interactive Analysis", last_dashboard_frame)

    # --- INPUT HANDLING ---
    wait_time = 30 if paused else 1
    key = cv2.waitKey(wait_time) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused
    elif key == ord('r'):
        # DRAW ROIs
        if current_raw_frame is not None:
            print(">>> DRAW MODE STARTED. Draw boxes. Press SPACE/ENTER to finish. ESC to cancel.")
            # We use current_raw_frame so user sees clean video
            rects = cv2.selectROIs("Draw Terminals (SPACE to save)", current_raw_frame, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow("Draw Terminals (SPACE to save)")
            
            # If user didn't cancel (returned valid tuple)
            if len(rects) > 0:
                rois = []
                for i, r in enumerate(rects):
                    rois.append({
                        "id": i+1,
                        "rect": tuple(map(int, r)), # Corrected for JSON serialization
                        "label": f"T{i+1}"
                    })
                
                # Save config
                with open(ROI_CONFIG_FILE, 'w') as f:
                    json.dump(rois, f)
                
                # Reset States
                roi_states = init_roi_states(rois)
                print(f">>> SAVED {len(rois)} ROIs. Analysis Reset.")
            else:
                print(">>> Draw Cancelled.")

if last_dashboard_frame is not None:
    cv2.imwrite(OUTPUT_IMAGE_PATH, last_dashboard_frame)

cap.release()
out.release()
cv2.destroyAllWindows()