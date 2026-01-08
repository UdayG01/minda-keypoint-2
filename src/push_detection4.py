import cv2
import numpy as np
import torch
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
ROI_CONFIG_FILE = "roi_config.json"

# Output Paths
OUTPUT_DIR = "analysis_outputs"
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "output_distance_check.mp4")
OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_DIR, "final_summary.png")

# Detection Constants
CONTACT_THRESHOLD = 23.5  # Pixels. Distance considered "Touching/Zero"
BUFFER_SIZE = 150         # Width of the graph in frames

# Layout Constants
TOTAL_WIDTH = 1280
TOTAL_HEIGHT = 720
PANEL_WIDTH = TOTAL_WIDTH // 2 

# ==============================
# SETUP
# ==============================
# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- CUDA / DEVICE CHECK ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("=================================================")
print(f"COMPUTE DEVICE: {device.upper()}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    # OPTIMIZATION: Enable CuDNN benchmark for consistent input sizes
    torch.backends.cudnn.benchmark = True
    print("OPTIMIZATION: Enabled CuDNN benchmark and FP16 (half-precision).")
else:
    print("WARNING: Running on CPU. Install pytorch-cuda for better performance.")
print("=================================================")

print(f"Loading model: {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
# Explicitly move model to device
model.to(device)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

# Video Writer Setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (TOTAL_WIDTH, TOTAL_HEIGHT))

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
            "status": "NG",          # NG (Red) -> OK (Green)
            "min_dist_seen": 9999.0, # Track closest approach
            "frame_ok": -1           # Frame number when it turned OK
        })
    return states

# Initial Load
rois = load_rois()
roi_states = init_roi_states(rois)

# Graph Buffer (Stores the minimum distance of the ACTIVE pair in current frame)
dist_buffer = deque([300.0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)

paused = False
frame_count = 0
current_raw_frame = None 
final_frame = None

print(f"PROCESSING: {VIDEO_PATH}")
print(f"SAVING TO:  {OUTPUT_VIDEO_PATH}")
print(f"LOGIC:      Distance < {CONTACT_THRESHOLD}px = PUSH (OK)")
print("=================================================")

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret: break
        current_raw_frame = frame.copy()
        frame_count += 1
        
        # --- 1. INFERENCE ---
        # OPTIMIZATION: half=True uses FP16 on GPU (much faster)
        # device=device ensures we stay on GPU
        results = model.track(
            frame, 
            persist=True, 
            verbose=False, 
            conf=0.10, 
            imgsz=960, 
            device=device, 
            half=(device == 'cuda')
        )
        
        curr_positions = {} # {id: (x, y)}
        
        if results[0].boxes.id is not None and results[0].keypoints is not None:
            # Transfer data to CPU for logic processing
            ids = results[0].boxes.id.int().cpu().tolist()
            kps = results[0].keypoints.xy.cpu().numpy()
            
            for obj_id, kp_set in zip(ids, kps):
                valid_kps = [k for k in kp_set if k[0] > 0 and k[1] > 0]
                if valid_kps:
                    curr_positions[obj_id] = valid_kps[0]

        # --- 2. IDENTIFY TERMINALS (Objects Inside ROIs) ---
        roi_terminals = {} # {roi_index: obj_id}
        used_ids = set()
        
        for i, r_state in enumerate(roi_states):
            rx, ry, rw, rh = r_state["data"]["rect"]
            roi_center = (rx + rw/2, ry + rh/2)
            
            best_id = None
            min_dist = 9999.0
            for oid, pos in curr_positions.items():
                d = math.hypot(pos[0] - roi_center[0], pos[1] - roi_center[1])
                if d < min_dist and d < max(rw, rh): 
                    min_dist = d
                    best_id = oid
            
            if best_id is not None:
                roi_terminals[i] = best_id
                used_ids.add(best_id)

        # --- 3. PAIRING & DISTANCE CALCULATION ---
        active_pairs = [] # [(term_pos, conn_pos, distance, roi_idx)]
        frame_min_distance = 300.0 # Default high value for graph
        
        for oid, c_pos in curr_positions.items():
            if oid in used_ids: continue # Skip terminals
            
            nearest_roi_idx = None
            min_dist_to_term = 9999.0
            nearest_term_pos = None
            
            for i, term_id in roi_terminals.items():
                t_pos = curr_positions[term_id]
                d = math.hypot(c_pos[0] - t_pos[0], c_pos[1] - t_pos[1])
                if d < min_dist_to_term:
                    min_dist_to_term = d
                    nearest_roi_idx = i
                    nearest_term_pos = t_pos
            
            if nearest_roi_idx is not None and min_dist_to_term < 400:
                active_pairs.append((nearest_term_pos, c_pos, min_dist_to_term, nearest_roi_idx))
                
                r_state = roi_states[nearest_roi_idx]
                if min_dist_to_term < r_state["min_dist_seen"]:
                    r_state["min_dist_seen"] = min_dist_to_term
                
                if min_dist_to_term < CONTACT_THRESHOLD:
                    if r_state["status"] == "NG":
                        print(f">>> CONTACT CONFIRMED: {r_state['data']['label']}")
                        r_state["status"] = "OK"
                        r_state["frame_ok"] = frame_count

        if active_pairs:
            active_pairs.sort(key=lambda x: x[2])
            frame_min_distance = active_pairs[0][2]
        
        dist_buffer.append(frame_min_distance)

        # --- 4. VISUALIZATION ---
        vis_frame = frame.copy()
        
        for r_state in roi_states:
            rx, ry, rw, rh = r_state["data"]["rect"]
            color = (0, 0, 255) # Red (NG)
            if r_state["status"] == "OK": color = (0, 255, 0) # Green (OK)
            
            cv2.rectangle(vis_frame, (rx, ry), (rx+rw, ry+rh), color, 2)
            cv2.putText(vis_frame, r_state['data']['label'], (rx, ry-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for (t_pos, c_pos, dist, r_idx) in active_pairs:
            cv2.circle(vis_frame, (int(t_pos[0]), int(t_pos[1])), 5, (255, 0, 255), -1)
            cv2.circle(vis_frame, (int(c_pos[0]), int(c_pos[1])), 5, (0, 255, 255), -1)
            
            line_color = (255, 255, 255)
            if dist < CONTACT_THRESHOLD: line_color = (0, 255, 0)
            
            cv2.line(vis_frame, (int(t_pos[0]), int(t_pos[1])), (int(c_pos[0]), int(c_pos[1])), line_color, 2)
            mid_x, mid_y = (t_pos[0]+c_pos[0])//2, (t_pos[1]+c_pos[1])//2
            cv2.putText(vis_frame, f"{int(dist)}px", (int(mid_x), int(mid_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 1)

        last_annotated_frame = vis_frame

    else:
        if current_raw_frame is not None:
            view = current_raw_frame.copy()
            cv2.putText(view, "PAUSED - Press 'r' to Draw ROIs", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            last_annotated_frame = view
        else:
            last_annotated_frame = np.zeros((720,1280,3), dtype=np.uint8)

    # --- DASHBOARD & SAVE ---
    video_panel = cv2.resize(last_annotated_frame, (PANEL_WIDTH, TOTAL_HEIGHT))
    dashboard = np.full((TOTAL_HEIGHT, PANEL_WIDTH, 3), (20, 20, 20), dtype=np.uint8)
    
    cv2.putText(dashboard, "TERMINAL STATUS", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    for i, r_state in enumerate(roi_states):
        y = 80 + (i * 40)
        status = r_state["status"]
        col = (0, 255, 0) if status == "OK" else (0, 0, 255)
        cv2.putText(dashboard, f"{r_state['data']['label']}: {status}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

    margin = 40
    graph_top = 300
    graph_bottom = TOTAL_HEIGHT - 50
    graph_h = graph_bottom - graph_top
    
    cv2.rectangle(dashboard, (margin, graph_top), (PANEL_WIDTH-margin, graph_bottom), (30, 30, 30), -1)
    cv2.putText(dashboard, "PROXIMITY (Distance to Target)", (margin, graph_top-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    thresh_y = graph_bottom - int((CONTACT_THRESHOLD / 300.0) * graph_h)
    cv2.line(dashboard, (margin, thresh_y), (PANEL_WIDTH-margin, thresh_y), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(dashboard, "CONTACT THRESHOLD", (margin+5, thresh_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    if len(dist_buffer) > 1:
        pts = []
        for i, val in enumerate(dist_buffer):
            x = margin + int(i * (PANEL_WIDTH - 2*margin) / BUFFER_SIZE)
            norm_val = min(val, 300.0) / 300.0
            y = graph_bottom - int(norm_val * graph_h)
            pts.append((x, y))
        cv2.polylines(dashboard, [np.array(pts)], False, (0, 255, 255), 2, cv2.LINE_AA)

    final_frame = np.hstack((video_panel, dashboard))
    
    # WRITE TO VIDEO FILE
    out.write(final_frame)
    cv2.imshow("Distance Logic", final_frame)

    key = cv2.waitKey(30 if paused else 1) & 0xFF
    if key == ord('q'): break
    elif key == ord('p'): paused = not paused
    elif key == ord('r'):
        if current_raw_frame is not None:
            rects = cv2.selectROIs("Define Terminals", current_raw_frame, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow("Define Terminals")
            if len(rects) > 0:
                rois = [{"id": i+1, "rect": tuple(map(int, r)), "label": f"T{i+1}"} for i, r in enumerate(rects)]
                with open(ROI_CONFIG_FILE, 'w') as f: json.dump(rois, f)
                roi_states = init_roi_states(rois)
                dist_buffer = deque([300.0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)

# Cleanup
if final_frame is not None:
    cv2.imwrite(OUTPUT_IMAGE_PATH, final_frame)
cap.release()
out.release()
cv2.destroyAllWindows()
print(f">>> SAVED VIDEO TO: {OUTPUT_VIDEO_PATH}")