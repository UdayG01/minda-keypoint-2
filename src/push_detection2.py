import cv2
import numpy as np
import torch
try:
    import intel_extension_for_pytorch as ipex
    HAS_IPEX = True
except ImportError:
    HAS_IPEX = False
from ultralytics import YOLO
from collections import deque
import math
import json
import os

# ==============================
# CONFIGURATION
# ==============================
VIDEO_PATH = "test_video/2025-10-24 13-09-37.mkv"
MODEL_PATH = "model/best.pt"
OUTPUT_VIDEO_PATH = "analysis_outputs/analysis_visual2.mp4"
OUTPUT_IMAGE_PATH = "analysis_outputs/analysis_summary2.png"
ROI_CONFIG_FILE = "roi_config.json"

# Analysis Constants
BUFFER_SIZE = 100        # Frame history for graphs
PUSH_SPEED_THRESH = 3.0 # Speed at which distance shrinks (pixels/frame)
PROXIMITY_THRESH = 40.0  # Max distance to validate the push
PUSH_COOLDOWN = 5       # Frames to wait before counting a second push

# Layout Constants
TOTAL_WIDTH = 1280
TOTAL_HEIGHT = 720
PANEL_WIDTH = TOTAL_WIDTH // 2 

# ==============================
# SETUP
# ==============================
print(f"Loading model: {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

# Device selection for IPEX/XPU
device = "cpu"
if HAS_IPEX:
    try:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            print("Detecting Intel XPU...")
            device = "xpu"
            print(f"Switching to XPU: {torch.xpu.get_device_name(0)}")
            # Optional: Optimize model with IPEX
            # model.model = ipex.optimize(model.model)
    except Exception as e:
        print(f"Failed to initialize XPU: {e}")

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
            "status": "NG",
            "last_push_frame": -999
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
push_events = [] 

roi_dist_history = {} # { roi_index: (frame_number, distance_value) }

paused = False
frame_count = 0
last_dashboard_frame = None
current_raw_frame = None 

print("=================================================")
print("CONTROLS:")
print("  'p' : Pause / Resume video")
print("  'r' : Draw ROIs (Define Terminals)")
print("  'q' : Quit")
print("=================================================")

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break
        current_raw_frame = frame.copy()
        frame_count += 1
        
        # --- 1. INFERENCE ---
        results = model.track(frame, persist=True, verbose=False, conf=0.35, imgsz=960, device=device)
        
        curr_positions = {}
        curr_bboxes = {} # Store bounding boxes for visualization
        
        if results[0].boxes.id is not None and results[0].keypoints is not None:
            ids = results[0].boxes.id.int().cpu().tolist()
            kps = results[0].keypoints.xy.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            for obj_id, kp_set, box in zip(ids, kps, boxes):
                valid_kps = [k for k in kp_set if k[0] > 0 and k[1] > 0]
                if valid_kps:
                    curr_positions[obj_id] = valid_kps[0]
                    curr_bboxes[obj_id] = box

        # --- 2. LOGIC: STRICT PAIRING ---
        
        # Step A: Identify Stationary "Terminal Objects" inside ROIs
        roi_terminals = {} 
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

        # Step B: Identify "Free Connectors" and assign to NEAREST Terminal only
        roi_assignments = {i: [] for i in range(len(roi_states))}
        
        for oid, pos in curr_positions.items():
            if oid in used_ids: continue # Is terminal
            
            nearest_roi_idx = None
            min_dist_to_term = 9999.0
            
            for i, term_id in roi_terminals.items():
                t_pos = curr_positions[term_id]
                d = math.hypot(pos[0] - t_pos[0], pos[1] - t_pos[1])
                if d < min_dist_to_term:
                    min_dist_to_term = d
                    nearest_roi_idx = i
            
            if nearest_roi_idx is not None:
                roi_assignments[nearest_roi_idx].append((oid, min_dist_to_term))

        # --- 3. PROCESS PUSH LOGIC ---
        max_closing_speed = 0.0 
        min_proximity = 300.0   
        active_pairs = [] 
        
        for i, r_state in enumerate(roi_states):
            current_dist = 9999.0
            closing_speed = 0.0
            assigned = roi_assignments[i]
            
            if assigned and i in roi_terminals:
                assigned.sort(key=lambda x: x[1]) 
                connector_id, dist = assigned[0]
                current_dist = dist
                
                term_pos = curr_positions[roi_terminals[i]]
                conn_pos = curr_positions[connector_id]
                active_pairs.append((term_pos, conn_pos))

                if current_dist < min_proximity: min_proximity = current_dist
                
                if i in roi_dist_history:
                    last_frame_idx, last_dist = roi_dist_history[i]
                    if frame_count - last_frame_idx == 1:
                        closing_speed = last_dist - current_dist
                
                roi_dist_history[i] = (frame_count, current_dist)
                if closing_speed > max_closing_speed: max_closing_speed = closing_speed

                if closing_speed > PUSH_SPEED_THRESH:
                    if (frame_count - r_state["last_push_frame"]) > PUSH_COOLDOWN:
                        if current_dist < PROXIMITY_THRESH:
                            r_state["last_push_frame"] = frame_count
                            r_state["pushes_detected"] += 1
                            if r_state["pushes_detected"] >= 2:
                                r_state["status"] = "OK"
                            push_events.append((frame_count, r_state['data']['label']))
                            print(f">>> PUSH on {r_state['data']['label']}")

        speed_buffer.append(max_closing_speed)
        dist_buffer.append(min_proximity)
        
        # --- DRAW DETECTIONS (BOXES & KEYPOINTS) ---
        vis_frame = frame.copy()
        
        for oid, box in curr_bboxes.items():
            x1, y1, x2, y2 = map(int, box)
            
            # Determine color: Terminal (Purple) or Connector (Blue)
            is_terminal = oid in roi_terminals.values()
            
            color = (255, 150, 0) # Blue (Connector)
            label = f"ID:{oid}"
            
            if is_terminal:
                color = (200, 50, 200) # Purple (Terminal)
                label = f"T-ID:{oid}"

            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        for oid, pos in curr_positions.items():
            cv2.circle(vis_frame, (int(pos[0]), int(pos[1])), 4, (0, 255, 255), -1) # Yellow Dot

        last_annotated_frame = vis_frame

    else:
        if current_raw_frame is not None:
            last_annotated_frame = cv2.addWeighted(current_raw_frame, 0.7, np.zeros_like(current_raw_frame), 0.3, 0)
            cv2.putText(last_annotated_frame, "PAUSED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 200), 5)
            cv2.putText(last_annotated_frame, "Press 'r' to Define Terminals", (55, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            last_annotated_frame = np.zeros((TOTAL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)

    # --- VISUALIZATION OUTPUT ---
    video_view = last_annotated_frame.copy()
    
    # Draw ROIs
    for r_state in roi_states:
        rx, ry, rw, rh = r_state["data"]["rect"]
        color = (0, 0, 255) 
        if r_state["status"] == "OK": color = (0, 255, 0)
        elif r_state["pushes_detected"] == 1: color = (0, 255, 255)
        cv2.rectangle(video_view, (rx, ry), (rx+rw, ry+rh), color, 2)
        cv2.putText(video_view, f"{r_state['data']['label']} ({r_state['pushes_detected']})", (rx, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    # Draw Active Pairings
    if not paused:
        for (p1, p2) in active_pairs:
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
            cv2.line(video_view, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)
            mid = ((pt1[0]+pt2[0])//2, (pt1[1]+pt2[1])//2)
            cv2.circle(video_view, mid, 2, (0, 255, 255), -1)

    video_panel = cv2.resize(video_view, (PANEL_WIDTH, TOTAL_HEIGHT))
    
    # Dashboard
    if roi_states:
        dash_w, dash_h = 240, 50 + (len(roi_states) * 35)
        dx, dy = PANEL_WIDTH - dash_w - 20, 20
        overlay = video_panel.copy()
        cv2.rectangle(overlay, (dx, dy), (dx+dash_w, dy+dash_h), (10, 10, 10), -1)
        video_panel = cv2.addWeighted(overlay, 0.7, video_panel, 0.3, 0)
        cv2.putText(video_panel, "TERMINAL STATUS", (dx+10, dy+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        for i, r_state in enumerate(roi_states):
            y = dy + 70 + (i * 35)
            col = (100, 100, 255) if r_state["status"] == "NG" else (100, 255, 100)
            cv2.putText(video_panel, f"{r_state['data']['label']}: {r_state['status']}", (dx+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            for p in range(2):
                dot_col = col if p < r_state["pushes_detected"] else (50, 50, 50)
                cv2.circle(video_panel, (dx + 200 + (p*15), y-5), 5, dot_col, -1)

    # Graphs
    dashboard = np.full((TOTAL_HEIGHT, PANEL_WIDTH, 3), (20, 20, 20), dtype=np.uint8)
    margin = 40
    gh = (TOTAL_HEIGHT - 3*margin) // 2
    g1t, g1b = margin, margin + gh
    g2t, g2b = g1b + margin, g1b + margin + gh

    cv2.rectangle(dashboard, (margin, g1t), (PANEL_WIDTH-margin, g1b), (35, 35, 35), -1)
    cv2.putText(dashboard, "CLOSING SPEED", (margin, g1t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
    if len(speed_buffer) > 1:
        pts = []
        for i, val in enumerate(speed_buffer):
            x = margin + int(i*(PANEL_WIDTH-2*margin)/BUFFER_SIZE)
            y = g1b - int(min(val, 50.0)/50.0 * gh)
            pts.append((x,y))
        cv2.polylines(dashboard, [np.array(pts)], False, (0,255,255), 2, cv2.LINE_AA)
        th_y = g1b - int((PUSH_SPEED_THRESH/50.0)*gh)
        cv2.line(dashboard, (margin, th_y), (PANEL_WIDTH-margin, th_y), (0,0,150), 1)

    cv2.rectangle(dashboard, (margin, g2t), (PANEL_WIDTH-margin, g2b), (35, 35, 35), -1)
    cv2.putText(dashboard, "PROXIMITY", (margin, g2t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
    if len(dist_buffer) > 1:
        pts = []
        for i, val in enumerate(dist_buffer):
            x = margin + int(i*(PANEL_WIDTH-2*margin)/BUFFER_SIZE)
            y = g2b - int(min(val, 300.0)/300.0 * gh)
            pts.append((x,y))
        cv2.polylines(dashboard, [np.array(pts)], False, (0,255,0), 2, cv2.LINE_AA)
        th_y2 = g2b - int((PROXIMITY_THRESH/300.0)*gh)
        cv2.line(dashboard, (margin, th_y2), (PANEL_WIDTH-margin, th_y2), (0,150,0), 1)

    for (p_frame, p_label) in push_events:
        frames_ago = frame_count - p_frame
        if frames_ago < BUFFER_SIZE and frames_ago >= 0:
            x = PANEL_WIDTH - margin - int(frames_ago * (PANEL_WIDTH-2*margin)/BUFFER_SIZE)
            cv2.line(dashboard, (x, g1t), (x, g2b), (255, 255, 255), 1)
            cv2.putText(dashboard, "PUSH", (x-20, g1t+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    final_frame = np.hstack((video_panel, dashboard))
    out.write(final_frame)
    cv2.imshow("Interactive Analysis", final_frame)

    wait_time = 30 if paused else 1
    key = cv2.waitKey(wait_time) & 0xFF
    if key == ord('q'): break
    elif key == ord('p'): paused = not paused
    elif key == ord('r'):
        if current_raw_frame is not None:
            print(">>> DRAW MODE. SPACE to save.")
            rects = cv2.selectROIs("Draw Terminals", current_raw_frame, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow("Draw Terminals")
            if len(rects) > 0:
                rois = [{"id": i+1, "rect": tuple(map(int, r)), "label": f"T{i+1}"} for i, r in enumerate(rects)]
                with open(ROI_CONFIG_FILE, 'w') as f: json.dump(rois, f)
                roi_states = init_roi_states(rois)
                roi_dist_history = {}
                push_events = [] 
                print(f">>> SAVED {len(rois)} ROIs.")
            else: print(">>> Draw Cancelled.")

if final_frame is not None: cv2.imwrite(OUTPUT_IMAGE_PATH, final_frame)
cap.release()
out.release()
cv2.destroyAllWindows()