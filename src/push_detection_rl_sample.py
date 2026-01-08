import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import math
import json
import os
import joblib
from sklearn.ensemble import RandomForestClassifier

# ==============================
# CONFIGURATION
# ==============================
# CHANGE THIS PATH TO TRAIN ON DIFFERENT VIDEOS
VIDEO_PATH = "test_video/2025-10-24 13-09-37.mkv" 

MODEL_PATH = "model/best.pt"
ROI_CONFIG_FILE = "roi_config.json"

# PERSISTENT STORAGE (This enables iterative training across videos)
DATASET_FILE = "push_dataset.npz"
ML_MODEL_FILE = "push_classifier.pkl"

# Analysis Constants
WINDOW_SIZE = 15         # Feature Window (Must stay consistent across videos)
PREDICTION_CONFIDENCE = 0.7 

# Layout Constants
TOTAL_WIDTH = 1280
TOTAL_HEIGHT = 720
PANEL_WIDTH = TOTAL_WIDTH // 2 

# ==============================
# ML / CLASSIFIER UTILS
# ==============================
class PushDetector:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.X_data = [] 
        self.y_data = [] 
        
        # 1. Load Existing Model (The "Brain")
        if os.path.exists(ML_MODEL_FILE):
            print(f">>> Loading existing brain from {ML_MODEL_FILE}...")
            try:
                self.model = joblib.load(ML_MODEL_FILE)
                self.is_trained = True
            except Exception as e:
                print(f"Error loading model: {e}")
        
        # 2. Load Existing Dataset (The "Memory")
        if os.path.exists(DATASET_FILE):
            print(f">>> Loading training history from {DATASET_FILE}...")
            try:
                data = np.load(DATASET_FILE)
                loaded_X = data['X']
                loaded_y = data['y']
                
                # Validation: Ensure feature shape matches current config
                if len(loaded_X) > 0 and loaded_X.shape[1] != WINDOW_SIZE * 2:
                    print("!!! WARNING: Saved dataset has different feature shape. Starting fresh.")
                else:
                    self.X_data = list(loaded_X)
                    self.y_data = list(loaded_y)
                    print(f">>> SUCCESSFULLY LOADED {len(self.y_data)} historical samples.")
            except Exception as e:
                print(f"Error loading dataset: {e}")

    def extract_features(self, dist_history, speed_history):
        """ Features: [normalized_dist_history, raw_speed_history] """
        if len(dist_history) < WINDOW_SIZE: return None
        
        d = np.array(list(dist_history))[-WINDOW_SIZE:]
        s = np.array(list(speed_history))[-WINDOW_SIZE:]
        
        # Normalize distance to be scale-invariant (0.0 to 1.0 based on max in window)
        d_norm = d / (np.max(d) + 1e-5) 
        
        return np.concatenate([d_norm, s])

    def add_sample(self, features, label):
        self.X_data.append(features)
        self.y_data.append(label)

    def train(self):
        if len(self.y_data) < 10:
            print(">>> Not enough data to train (need at least 10 samples).")
            return
        
        print(f">>> Training on TOTAL {len(self.y_data)} samples (Historical + New)...")
        X = np.array(self.X_data)
        y = np.array(self.y_data)
        
        # Random Forest is excellent for this: handles noise well, doesn't overfit easily
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True
        
        # Save both model and dataset to disk for next time
        joblib.dump(self.model, ML_MODEL_FILE)
        np.savez(DATASET_FILE, X=X, y=y)
        print(f">>> SAVED model and {len(y)} samples to disk.")

    def predict(self, features):
        if not self.is_trained or self.model is None: return 0.0
        return self.model.predict_proba([features])[0][1] 

# ==============================
# SETUP
# ==============================
print(f"Loading YOLO: {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
detector = PushDetector() # Automatically loads history

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

# Load ROIs
rois = []
if os.path.exists(ROI_CONFIG_FILE):
    with open(ROI_CONFIG_FILE, 'r') as f: rois = json.load(f)

# Initialize States
roi_states = []
for i, r in enumerate(rois):
    roi_states.append({
        "data": r,
        "pushes_detected": 0,
        "status": "NG",
        "last_push_frame": -999,
        "hist_dist": deque([100.0]*WINDOW_SIZE, maxlen=WINDOW_SIZE),
        "hist_speed": deque([0.0]*WINDOW_SIZE, maxlen=WINDOW_SIZE)
    })

speed_buffer = deque([0.0] * 100, maxlen=100)
ml_prob_buffer = deque([0.0] * 100, maxlen=100) 
events = []

paused = False
frame_count = 0

print("=================================================")
print(f"PROCESSING: {VIDEO_PATH}")
print(f"HISTORY:    {len(detector.y_data)} samples loaded")
print("=================================================")
print("CONTROLS:")
print("  'p'     : Pause / Resume")
print("  'SPACE' : (Paused) TEACH -> Label as PUSH")
print("  'n'     : (Paused) TEACH -> Label as BACKGROUND")
print("  't'     : TRAIN & SAVE (Persist data to disk)")
print("  'q'     : QUIT")
print("=================================================")

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        # 1. TRACKING
        results = model.track(frame, persist=True, verbose=False, conf=0.35, imgsz=960)
        
        curr_positions = {}
        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.int().cpu().tolist()
            kps = results[0].keypoints.xy.cpu().numpy()
            for obj_id, kp_set in zip(ids, kps):
                valid_kps = [k for k in kp_set if k[0] > 0 and k[1] > 0]
                if valid_kps: curr_positions[obj_id] = valid_kps[0]

        # 2. STRICT PAIRING LOGIC
        roi_terminals = {}
        used_ids = set()
        for i, r_state in enumerate(roi_states):
            rx, ry, rw, rh = r_state["data"]["rect"]
            roi_center = (rx + rw/2, ry + rh/2)
            best_id, min_d = None, 9999.0
            for oid, pos in curr_positions.items():
                d = math.hypot(pos[0] - roi_center[0], pos[1] - roi_center[1])
                if d < min_d and d < max(rw,rh):
                    min_d = d; best_id = oid
            if best_id is not None:
                roi_terminals[i] = best_id
                used_ids.add(best_id)

        roi_assignments = {i: [] for i in range(len(roi_states))}
        for oid, pos in curr_positions.items():
            if oid in used_ids: continue
            nearest_roi, min_d = None, 9999.0
            for i, term_id in roi_terminals.items():
                t_pos = curr_positions[term_id]
                d = math.hypot(pos[0] - t_pos[0], pos[1] - t_pos[1])
                if d < min_d: min_d = d; nearest_roi = i
            if nearest_roi is not None:
                roi_assignments[nearest_roi].append((oid, min_d))

        # 3. FEATURE EXTRACTION & INFERENCE
        max_prob = 0.0
        
        for i, r_state in enumerate(roi_states):
            current_dist = 200.0
            assigned = roi_assignments[i]
            
            if assigned and i in roi_terminals:
                assigned.sort(key=lambda x: x[1])
                conn_id, dist = assigned[0]
                current_dist = dist
                # Visual Line
                t_pos = curr_positions[roi_terminals[i]]
                c_pos = curr_positions[conn_id]
                cv2.line(frame, (int(t_pos[0]), int(t_pos[1])), (int(c_pos[0]), int(c_pos[1])), (255,255,255), 1)

            last_dist = r_state["hist_dist"][-1]
            current_speed = last_dist - current_dist
            
            r_state["hist_dist"].append(current_dist)
            r_state["hist_speed"].append(current_speed)
            
            features = detector.extract_features(r_state["hist_dist"], r_state["hist_speed"])
            
            if features is not None:
                r_state["current_features"] = features 
                
                # PREDICT
                if detector.is_trained:
                    prob = detector.predict(features)
                    if prob > max_prob: max_prob = prob
                    
                    if prob > PREDICTION_CONFIDENCE:
                        if (frame_count - r_state["last_push_frame"]) > 25:
                            r_state["last_push_frame"] = frame_count
                            r_state["pushes_detected"] += 1
                            if r_state["pushes_detected"] >= 2: r_state["status"] = "OK"
                            events.append((frame_count, f"{r_state['data']['label']}"))
                            print(f">>> PUSH DETECTED: {r_state['data']['label']} (Confidence: {prob:.2f})")

        ml_prob_buffer.append(max_prob)
        last_annotated_frame = frame.copy()

    else:
        # === TEACHING UI ===
        view = current_raw_frame.copy() if 'current_raw_frame' in locals() else np.zeros((720,1280,3), dtype=np.uint8)
        
        # Darken for readability
        view = cv2.addWeighted(view, 0.3, np.zeros_like(view), 0.7, 0)
        
        cv2.putText(view, f"TEACH MODE (Samples: {len(detector.y_data)})", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
        
        cv2.putText(view, "Did a PUSH happen just now?", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(view, "[SPACE] YES - It was a Push", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(view, "[N]     NO  - It was Background", (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.putText(view, "-----------------------------", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 1)
        cv2.putText(view, "[T]     TRAIN & SAVE to Disk", (80, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 255), 2)
        cv2.putText(view, "[P]     Resume Video", (80, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        last_annotated_frame = view

    # --- DASHBOARD ---
    video_panel = cv2.resize(last_annotated_frame, (PANEL_WIDTH, TOTAL_HEIGHT))
    dashboard = np.full((TOTAL_HEIGHT, PANEL_WIDTH, 3), (20, 20, 20), dtype=np.uint8)
    
    margin = 40
    gh = (TOTAL_HEIGHT - 3*margin) // 2
    g1t, g1b = margin, margin + gh
    
    # Confidence Plot
    cv2.rectangle(dashboard, (margin, g1t), (PANEL_WIDTH-margin, g1b), (35, 35, 35), -1)
    cv2.putText(dashboard, "AI CONFIDENCE", (margin, g1t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 1)
    if len(ml_prob_buffer) > 1:
        pts = []
        for i, val in enumerate(ml_prob_buffer):
            x = margin + int(i*(PANEL_WIDTH-2*margin)/100)
            y = g1b - int(val * gh)
            pts.append((x,y))
        cv2.polylines(dashboard, [np.array(pts)], False, (255,0,255), 2, cv2.LINE_AA)
        cv2.line(dashboard, (margin, g1b-int(PREDICTION_CONFIDENCE*gh)), (PANEL_WIDTH-margin, g1b-int(PREDICTION_CONFIDENCE*gh)), (100,100,100), 1)

    for (p_frame, p_label) in events:
        frames_ago = frame_count - p_frame
        if frames_ago < 100 and frames_ago >= 0:
            x = PANEL_WIDTH - margin - int(frames_ago * (PANEL_WIDTH-2*margin)/100)
            cv2.line(dashboard, (x, g1t), (x, g1b), (255, 255, 255), 1)
            cv2.putText(dashboard, "PUSH", (x-10, g1t+15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    final_frame = np.hstack((video_panel, dashboard))
    cv2.imshow("Interactive Trainer", final_frame)

    key = cv2.waitKey(30 if paused else 1) & 0xFF
    if key == ord('q'): break
    elif key == ord('p'): paused = not paused
    elif key == ord('t') and paused:
        detector.train()
        paused = False 
    elif key == ord(' ') and paused: 
        print(">>> TEACHING: PUSH")
        for r in roi_states:
             if 'current_features' in r and r['current_features'] is not None:
                 detector.add_sample(r["current_features"], 1)
        paused = False 
    elif key == ord('n') and paused:
        print(">>> TEACHING: BACKGROUND")
        for r in roi_states:
             if 'current_features' in r and r['current_features'] is not None:
                 detector.add_sample(r["current_features"], 0)
        paused = False

cap.release()
cv2.destroyAllWindows()