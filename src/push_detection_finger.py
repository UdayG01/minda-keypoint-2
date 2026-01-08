"""
In this file, we use finger validation for both the pushes
"""


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
VIDEO_PATH = "test_video/2025-10-24 13-07-23.mkv"
MODEL_PATH = "model/best.pt"
OUTPUT_VIDEO_PATH = "analysis_outputs/analysis_visual2.mp4"
OUTPUT_IMAGE_PATH = "analysis_outputs/analysis_summary2.png"
ROI_CONFIG_FILE = "roi_config.json"

# Analysis Constants
BUFFER_SIZE = 100        # Frame history for graphs
PUSH_SPEED_THRESH = 3.0 # Speed at which distance shrinks (pixels/frame)
FINGER_MOVE_THRESH = 1.0 # Speed threshold for finger movement
PROXIMITY_THRESH = 40.0  # Max distance to validate the push
FINGER_PROXIMITY_THRESH = 75.0 # Max distance for finger to be considered "interacting"
PUSH_COOLDOWN = 3       # Frames to wait before counting a second push
CONNECTOR_MEMORY = 50   # Frames to remember a connector (approx 1s) for 1st push validation

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

roi_dist_history = {} # Still kept for backward compat if needed, but logic uses model.roi_...

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
            clss = results[0].boxes.cls.int().cpu().tolist()
            kps = results[0].keypoints.xy.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            for obj_id, cls_id, kp_set, box in zip(ids, clss, kps, boxes):
                valid_kps = [k for k in kp_set if k[0] > 0 and k[1] > 0]
                if valid_kps:
                    curr_positions[obj_id] = valid_kps[0]
                    curr_bboxes[obj_id] = (box, cls_id)

        # --- 2. LOGIC: STRICT PAIRING ---
        
        # Step A: Identify Stationary "Terminal Objects" inside ROIs (Classes 1, 2, 3)
        roi_terminals = {} 
        used_ids = set()
        
        for i, r_state in enumerate(roi_states):
            rx, ry, rw, rh = r_state["data"]["rect"]
            roi_center = (rx + rw/2, ry + rh/2)
            
            best_id = None
            min_dist = 9999.0
            for oid, pos in curr_positions.items():
                # Check class if available
                if oid in curr_bboxes:
                    cls_id = curr_bboxes[oid][1]
                    if cls_id not in [1, 2, 3]: # Strictly T1, T2, T3
                         continue

                d = math.hypot(pos[0] - roi_center[0], pos[1] - roi_center[1])
                if d < min_dist and d < max(rw, rh): 
                    min_dist = d
                    best_id = oid
            
            if best_id is not None:
                roi_terminals[i] = best_id
                used_ids.add(best_id)

        # Step B: Identify "Free Connectors" (Class 0) and "Fingers" (Class 4)
        roi_assignments = {i: [] for i in range(len(roi_states))}
        finger_assignments = {i: [] for i in range(len(roi_states))}
        
        for oid, pos in curr_positions.items():
            if oid in used_ids: continue # Is terminal
            
            if oid not in curr_bboxes: continue
            cls_id = curr_bboxes[oid][1]
            
            # Find nearest Terminal for this object
            nearest_roi_idx = None
            min_dist_to_term = 9999.0
            
            for i, term_id in roi_terminals.items():
                t_pos = curr_positions[term_id]
                d = math.hypot(pos[0] - t_pos[0], pos[1] - t_pos[1])
                if d < min_dist_to_term:
                    min_dist_to_term = d
                    nearest_roi_idx = i
            
            if nearest_roi_idx is not None:
                if cls_id == 0: # Connector (c1)
                    roi_assignments[nearest_roi_idx].append((oid, min_dist_to_term))
                elif cls_id == 4: # Finger
                    finger_assignments[nearest_roi_idx].append((oid, min_dist_to_term))

        # History State (Separate to prevent jumps when switching objects)
        if not hasattr(model, 'roi_conn_history'): model.roi_conn_history = {}
        if not hasattr(model, 'roi_finger_history'): model.roi_finger_history = {}
        if not hasattr(model, 'roi_conn_last_seen'): model.roi_conn_last_seen = {} # {roi_idx: frame_num}

        # --- 3. PROCESS PUSH LOGIC ---
        max_closing_speed = 0.0 
        min_proximity = 300.0   
        active_pairs = [] 
        
        for i, r_state in enumerate(roi_states):
            term_pos = None
            if i in roi_terminals:
                term_pos = curr_positions[roi_terminals[i]]
            
            if term_pos is None: continue

            # --- A. CONNECTOR ANALYSIS ---
            conn_trigger = False
            conn_dist = 300.0
            conn_speed = 0.0
            active_conn_id = None
            
            if roi_assignments[i]:
                # Update Memory (We saw a connector here)
                model.roi_conn_last_seen[i] = frame_count

                roi_assignments[i].sort(key=lambda x: x[1])
                c_id, c_dist = roi_assignments[i][0]
                active_conn_id = c_id
                conn_dist = c_dist
                
                # Speed Calc
                if i in model.roi_conn_history:
                    last_frame, last_dist = model.roi_conn_history[i]
                    if frame_count - last_frame == 1:
                        conn_speed = last_dist - conn_dist
                model.roi_conn_history[i] = (frame_count, conn_dist)
                
                # Trigger Check
                if conn_speed > PUSH_SPEED_THRESH and conn_dist < PROXIMITY_THRESH:
                    conn_trigger = True

                # Visualization Pair
                active_pairs.append((term_pos, curr_positions[c_id]))

            # --- B. FINGER ANALYSIS ---
            finger_trigger = False
            finger_dist = 300.0
            finger_speed = 0.0
            active_finger_id = None
            
            if finger_assignments[i]: # Track finger always (needed for 1st push occlusions)
                finger_assignments[i].sort(key=lambda x: x[1])
                f_id, f_dist = finger_assignments[i][0]
                active_finger_id = f_id
                finger_dist = f_dist
                
                # Speed Calc
                last_frame = -999
                if i in model.roi_finger_history:
                    last_frame, last_dist = model.roi_finger_history[i]
                    frame_diff = frame_count - last_frame
                    
                    # Relaxed check: Allow up to 5 frames gap (flicker protection)
                    if 0 < frame_diff <= 5: 
                        raw_speed = last_dist - finger_dist
                        finger_speed = raw_speed / frame_diff # Normalize speed
                        
                        # DEBUG PRINT
                        print(f"ROI {i}: Finger Speed: {finger_speed:.2f} (Dist: {finger_dist:.1f}) last_seen_conn: {frame_count - model.roi_conn_last_seen.get(i, -999)}")

                model.roi_finger_history[i] = (frame_count, finger_dist)
                
                # Trigger Check
                # FINGER LOGIC: Movement + Loose Proximity Zone
                if finger_speed > FINGER_MOVE_THRESH and finger_dist < FINGER_PROXIMITY_THRESH:
                    finger_trigger = True
                
                # Visualization Pair (Finger is Orange line)
                active_pairs.append((term_pos, curr_positions[f_id]))

            # --- C. DECISION LOGIC ---
            
            # Determine which metrics to show on graph (Min of active objects)
            current_dist = min(conn_dist, finger_dist)
            current_speed = max(conn_speed, finger_speed)
            
            if current_dist < min_proximity: min_proximity = current_dist
            if current_speed > max_closing_speed: max_closing_speed = current_speed

            # PUSH VALIDATION
            is_push = False
            trigger_source = ""
            
            if (frame_count - r_state["last_push_frame"]) > PUSH_COOLDOWN:
                if r_state["pushes_detected"] == 0:
                    # 1st Push: Connector OR (Finger + Valid Connector Memory)
                    if conn_trigger:
                        is_push = True
                        trigger_source = "CONNECTOR"
                    elif finger_trigger:
                        # Only allow finger if we saw a connector recently
                        last_seen = model.roi_conn_last_seen.get(i, -999)
                        if (frame_count - last_seen) < CONNECTOR_MEMORY:
                            is_push = True
                            trigger_source = "FINGER (Occluded)"
                
                elif r_state["pushes_detected"] >= 1:
                    # 2nd+ Push: Connector OR Finger
                    if conn_trigger:
                        is_push = True
                        trigger_source = "CONNECTOR"
                    elif finger_trigger:
                        is_push = True
                        trigger_source = "FINGER"
            else:
                 pass # Cooldown active

            # DEBUG DECISION LOG
            # if finger_trigger:
            #    print(f"DEBUG: ROI {i} Trig=True, Pushes={r_state['pushes_detected']}, CD_Check={(frame_count - r_state['last_push_frame']) > PUSH_COOLDOWN}, Mem={model.roi_conn_last_seen.get(i, -1)}")

            if is_push:
                r_state["last_push_frame"] = frame_count
                r_state["pushes_detected"] += 1
                
                if r_state["pushes_detected"] >= 2:
                    r_state["status"] = "OK"
                
                print(f">>> PUSH ({trigger_source}) on {r_state['data']['label']} (Total: {r_state['pushes_detected']})")
                push_events.append((frame_count, f"{r_state['data']['label']} ({trigger_source})"))
            
            elif finger_trigger and r_state["pushes_detected"] == 0:
                # Debug failure to enable 1st push
                last_seen = model.roi_conn_last_seen.get(i, -999)
                if (frame_count - last_seen) >= CONNECTOR_MEMORY:
                     # print(f"ROI {i}: Finger 1st Push IGNORED (No Connector Memory). Diff: {frame_count - last_seen}")
                     pass 

        speed_buffer.append(max_closing_speed)
        dist_buffer.append(min_proximity)
        
        # --- DRAW DETECTIONS (BOXES & KEYPOINTS) ---
        vis_frame = frame.copy()
        
        for oid, (box, cls_id) in curr_bboxes.items():
            x1, y1, x2, y2 = map(int, box)
            
            # Determine color: Terminal (Purple), Connector (Blue), Finger (Orange)
            color = (255, 150, 0) # Blue (Connector)
            label = f"C1:{oid}"
            
            if cls_id in [1, 2, 3]: # Terminals
                color = (200, 50, 200) # Purple
                label = f"T{cls_id}:{oid}"
            elif cls_id == 4: # Finger
                color = (0, 165, 255) # Orange
                label = f"Fgr:{oid}"

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
                # Reset States
                roi_states = init_roi_states(rois)
                if hasattr(model, 'roi_conn_history'): model.roi_conn_history = {}
                if hasattr(model, 'roi_finger_history'): model.roi_finger_history = {}
                if hasattr(model, 'roi_conn_last_seen'): model.roi_conn_last_seen = {}
                push_events = [] 
                print(f">>> SAVED {len(rois)} ROIs.")
            else: print(">>> Draw Cancelled.")

if final_frame is not None: cv2.imwrite(OUTPUT_IMAGE_PATH, final_frame)
cap.release()
out.release()
cv2.destroyAllWindows()