import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import math

# ==============================
# CONFIGURATION
# ==============================
VIDEO_PATH = "test_video/2025-10-24 13-09-37.mkv"
MODEL_PATH = "model/best.pt" 
OUTPUT_VIDEO_PATH = "analysis_outputs/analysis.mp4"
OUTPUT_IMAGE_PATH = "analysis_outputs/analysis_summary.png"

# Display Settings
# We set a fixed output resolution for a clean split-screen look
TOTAL_WIDTH = 1280       
TOTAL_HEIGHT = 720       
PANEL_WIDTH = TOTAL_WIDTH // 2  # Each half (Video / Graphs) gets 640px width

# Graph Analysis Settings
BUFFER_SIZE = 100        
PUSH_THRESHOLD = 4.0    

# Scale Settings
MAX_SPEED_Y = 50.0       
MAX_DIST_Y = 300.0       

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

# Output Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (TOTAL_WIDTH, TOTAL_HEIGHT))

# Data Buffers
speed_buffer = deque([0.0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
dist_buffer = deque([0.0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)

# Tracking State
object_positions = {} 

# Store the last frame of the dashboard to save as an image
last_dashboard_frame = None 

print("🚀 Starting Split-Screen Analysis... Press 'Q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. INFERENCE
    # Keep imgsz=960 for detection accuracy on small connectors
    results = model.track(frame, persist=True, verbose=False, conf=0.35, imgsz=960)
    
    # 2. PREPARE VIDEO PANEL (LEFT HALF)
    annotated_frame = results[0].plot(labels=False, conf=False)
    # Resize the high-res inference frame to fit the Left Panel (640x720)
    video_panel = cv2.resize(annotated_frame, (PANEL_WIDTH, TOTAL_HEIGHT))
    
    # 3. METRIC CALCULATION
    max_speed = 0.0
    min_distance = MAX_DIST_Y 
    
    current_frame_positions = {}
    active_obj_id = None

    if results[0].boxes.id is not None and results[0].keypoints is not None:
        ids = results[0].boxes.id.int().cpu().tolist()
        kps = results[0].keypoints.xy.cpu().numpy()
        
        for obj_id, kp_set in zip(ids, kps):
            valid_kps = [k for k in kp_set if k[0] > 0 and k[1] > 0]
            if valid_kps:
                cx, cy = valid_kps[0]
                current_frame_positions[obj_id] = (cx, cy)
                
                if obj_id in object_positions:
                    prev_x, prev_y = object_positions[obj_id]
                    speed = math.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)
                    if speed > max_speed:
                        max_speed = speed
                        active_obj_id = obj_id 
                
                object_positions[obj_id] = (cx, cy)

        if active_obj_id is not None and len(current_frame_positions) > 1:
            connector_pos = current_frame_positions[active_obj_id]
            distances = []
            for oid, pos in current_frame_positions.items():
                if oid != active_obj_id: 
                    d = math.sqrt((connector_pos[0] - pos[0])**2 + (connector_pos[1] - pos[1])**2)
                    distances.append(d)
            if distances:
                min_distance = min(distances)

    # Cleanup IDs
    current_ids = set(current_frame_positions.keys())
    for old_id in list(object_positions.keys()):
        if old_id not in current_ids:
            del object_positions[old_id]

    speed_buffer.append(max_speed)
    dist_buffer.append(min_distance)

    # 4. DASHBOARD PANEL (RIGHT HALF)
    dashboard = np.full((TOTAL_HEIGHT, PANEL_WIDTH, 3), (20, 20, 20), dtype=np.uint8)
    
    # --- LAYOUT CONSTANTS ---
    MARGIN = 30
    # Available height for graphs (total height minus margins)
    # We want 2 graphs stacked vertically with a gap in the middle
    available_h = TOTAL_HEIGHT - (4 * MARGIN) 
    single_graph_h = available_h // 2
    
    # --- GRAPH 1: VELOCITY (TOP) ---
    G1_X1, G1_Y1 = MARGIN, MARGIN
    G1_X2, G1_Y2 = PANEL_WIDTH - MARGIN, MARGIN + single_graph_h
    
    # Background Box
    cv2.rectangle(dashboard, (G1_X1, G1_Y1), (G1_X2, G1_Y2), (40, 40, 40), -1)
    cv2.putText(dashboard, "VELOCITY (Push Force)", (G1_X1 + 10, G1_Y1 + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Draw Velocity Plot
    if len(speed_buffer) > 1:
        pts = []
        for i, val in enumerate(speed_buffer):
            x = G1_X1 + int(i * (G1_X2 - G1_X1) / BUFFER_SIZE)
            norm_y = min(val, MAX_SPEED_Y) / MAX_SPEED_Y
            y = G1_Y2 - int(norm_y * single_graph_h)
            pts.append((x, y))
        cv2.polylines(dashboard, [np.array(pts)], False, (0, 255, 255), 2, cv2.LINE_AA)

    # Velocity Threshold Line
    thresh_y_local = G1_Y2 - int((PUSH_THRESHOLD/MAX_SPEED_Y) * single_graph_h)
    cv2.line(dashboard, (G1_X1, thresh_y_local), (G1_X2, thresh_y_local), (0, 0, 150), 1, cv2.LINE_AA)
    cv2.putText(dashboard, f"Threshold: {PUSH_THRESHOLD}", (G1_X2 - 120, thresh_y_local - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

    # --- SEPARATOR ---
    mid_y = TOTAL_HEIGHT // 2
    cv2.line(dashboard, (0, mid_y), (PANEL_WIDTH, mid_y), (100, 100, 100), 2)

    # --- GRAPH 2: PROXIMITY (BOTTOM) ---
    G2_X1, G2_Y1 = MARGIN, mid_y + MARGIN
    G2_X2, G2_Y2 = PANEL_WIDTH - MARGIN, mid_y + MARGIN + single_graph_h
    
    # Background Box
    cv2.rectangle(dashboard, (G2_X1, G2_Y1), (G2_X2, G2_Y2), (40, 40, 40), -1)
    cv2.putText(dashboard, "PROXIMITY (Distance)", (G2_X1 + 10, G2_Y1 + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw Distance Plot
    if len(dist_buffer) > 1:
        pts = []
        for i, val in enumerate(dist_buffer):
            x = G2_X1 + int(i * (G2_X2 - G2_X1) / BUFFER_SIZE)
            norm_y = min(val, MAX_DIST_Y) / MAX_DIST_Y
            y = G2_Y2 - int(norm_y * single_graph_h)
            pts.append((x, y))
        cv2.polylines(dashboard, [np.array(pts)], False, (0, 255, 0), 2, cv2.LINE_AA)
        
    cv2.putText(dashboard, f"Curr Dist: {min_distance:.1f}", (G2_X2 - 150, G2_Y1 + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 1)

    # --- STATUS OVERLAY ---
    # We display a large alert if push is detected
    if max_speed > PUSH_THRESHOLD:
        msg = "PUSH DETECTED"
        color = (0, 0, 255) # Red
        
        if min_distance < 50:
            msg = "CONNECTED"
            color = (0, 255, 0) # Green
            
        # Draw status box centered on the separator line
        box_w, box_h = 300, 60
        box_x = (PANEL_WIDTH - box_w) // 2
        box_y = mid_y - (box_h // 2)
        
        cv2.rectangle(dashboard, (box_x, box_y), (box_x + box_w, box_y + box_h), color, -1)
        cv2.rectangle(dashboard, (box_x, box_y), (box_x + box_w, box_y + box_h), (255,255,255), 2)
        
        text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = box_x + (box_w - text_size[0]) // 2
        text_y = box_y + (box_h + text_size[1]) // 2
        
        cv2.putText(dashboard, msg, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    last_dashboard_frame = dashboard.copy()

    # 5. COMBINE & DISPLAY
    combined = np.hstack((video_panel, dashboard))
    out.write(combined)
    cv2.imshow("Split-Screen Analysis", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- SAVE FINAL GRAPHS ---
if last_dashboard_frame is not None:
    cv2.imwrite(OUTPUT_IMAGE_PATH, last_dashboard_frame)
    print(f"✅ Saved final graph summary to {OUTPUT_IMAGE_PATH}")

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Saved video analysis to {OUTPUT_VIDEO_PATH}")