import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
import os
import sys
import warnings
import csv
import json
import re
import subprocess

# --- AI TRAINING DATA LOGGERS ---
CSV_FILE = "crash_training_data_v2.csv"  # V2 format!
EVENTS_FILE = "events_log.json"
TRACKING_FILE = "video_tracking.json"

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Track_ID", "Obj_Class", "BBox_CX", "BBox_CY", "BBox_W", "BBox_H", 
            "Scene_Density", "Ego_Drifting", "Distance_m", "Approach_Speed_kmh", 
            "Lateral_Speed_vx", "Vertical_Speed_vy", "TTC_sec", "Target_Correct",
            "Severity", "Start_Timing", "End_Quality", "Reward_Score"
        ])

try:
    import yt_dlp
except ImportError:
    print("yt-dlp not found! Please run: pip install yt-dlp")
    yt_dlp = None

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_GENERAL_PATH = "best.pt" 
MODEL_CAR_PATH = "selfdriving_openvino_model"       

HISTORY_LENGTH = 15  
LATERAL_THREAT_SPEED = 2.0 
REAL_WIDTHS = {'car': 2.0, 'truck': 2.5, 'bus': 3.0, 'default': 2.0}
FOCAL_CONSTANT = 1200 

# ==========================================
# STATE & FILE TRACKING UTILS (Synced with Colab V9.9)
# ==========================================
def load_json(filepath, default_val):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception:
            return default_val
    return default_val

def save_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def get_safe_filename(url_or_path):
    name = os.path.basename(url_or_path) if not url_or_path.startswith("http") else url_or_path
    safe = re.sub(r'[^a-zA-Z0-9]', '_', name)
    return safe[:40]

def audit_and_get_resume_state(video_path):
    """File Audit System: Synced with Colab V9.9 formatting."""
    tracking = load_json(TRACKING_FILE, {})
    safe_name = get_safe_filename(video_path)

    if video_path not in tracking:
        tracking[video_path] = {"version": 1, "completed_parts": [], "status": "processing"}

    state = tracking[video_path]
    
    if state.get("status") == "completed":
        state["version"] += 1
        state["completed_parts"] = []
        state["status"] = "processing"

    valid_parts =[]
    missing_gaps = []
    
    for p in state.get("completed_parts",[]):
        if os.path.exists(p['file']) and os.path.getsize(p['file']) > 1024:
            valid_parts.append(p)
        else:
            missing_gaps.append(p)

    state["completed_parts"] = valid_parts

    if missing_gaps:
        gap = missing_gaps[0]
        print(f"⚠️ Audit Alert: Missing part {gap['part']} for {safe_name}. Re-rendering {gap['start']}s to {gap['end']}s.")
        save_json(TRACKING_FILE, tracking)
        return gap['start'], gap['end'], gap['part'], gap['file']

    if valid_parts:
        last_part = max(valid_parts, key=lambda x: x['part'])
        next_part_num = last_part['part'] + 1
        start_sec = last_part['end']
    else:
        next_part_num = 1
        start_sec = 0.0

    output_filename = f"processed_{safe_name}_v{state['version']}_p{next_part_num}.mp4"
    save_json(TRACKING_FILE, tracking)
    
    return start_sec, None, next_part_num, output_filename

def update_tracking(video_path, part_num, start_sec, current_sec, output_filename, status="processing"):
    tracking = load_json(TRACKING_FILE, {})
    if video_path in tracking:
        state = tracking[video_path]
        parts = state.get("completed_parts",[])
        part_found = False
        for p in parts:
            if p['part'] == part_num:
                p['end'] = current_sec
                part_found = True
                break
        if not part_found:
            parts.append({'part': part_num, 'start': start_sec, 'end': current_sec, 'file': output_filename})
        state["completed_parts"] = parts
        state["status"] = status
        save_json(TRACKING_FILE, tracking)

def append_event_log(event_data):
    events = load_json(EVENTS_FILE,[])
    events.append(event_data)
    save_json(EVENTS_FILE, events)

# ==========================================
# IMAGE PROCESSING UTILS
# ==========================================
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def filter_colors(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    lower_white = np.array([0, 140, 0]); upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(hls, lower_white, upper_white)
    lower_yellow = np.array([10, 50, 90]); upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask=mask)

def canny(image):
    if image is None: return None
    darkened = adjust_gamma(image, gamma=0.7)
    contrast = enhance_contrast(darkened)
    filtered = filter_colors(contrast)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 40, 120)

def region_of_interest(image):
    height, width = image.shape[:2]
    polygons = np.array([[
        (int(width * 0.10), height),             
        (int(width * 0.90), height),             
        (int(width * 0.55), int(height * 0.65)), 
        (int(width * 0.45), int(height * 0.65))  
    ]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)

def make_coordinates(image, line_params):
    if line_params is None: return None
    slope, intercept = line_params
    y1 = image.shape[0]; y2 = int(y1 * 0.65)
    if abs(slope) < 1e-4: return None
    try: x1 = int((y1 - intercept)/slope); x2 = int((y2 - intercept)/slope); return np.array([x1, y1, x2, y2])
    except: return None

def average_slope_intercept(image, lines):
    left_fit, right_fit = [],[]
    if lines is None: return None, None
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2==x1: continue
        params = np.polyfit((x1, x2), (y1, y2), 1)
        slope = params[0]
        if slope < -0.4: left_fit.append(params)
        elif slope > 0.4: right_fit.append(params)
    l_avg = np.average(left_fit, axis=0) if left_fit else None
    r_avg = np.average(right_fit, axis=0) if right_fit else None
    return l_avg, r_avg

def get_lane_status(width, left_line, right_line):
    if left_line is None or right_line is None: return "LANE LOST", (0, 0, 255)
    left_x_bottom = left_line[0]; right_x_bottom = right_line[0]
    lane_center = (left_x_bottom + right_x_bottom) / 2
    image_center = width / 2
    deviation = image_center - lane_center
    threshold = width * 0.05
    if deviation > threshold: return "DRIFTING RIGHT >", (0, 165, 255)
    elif deviation < -threshold: return "< DRIFTING LEFT", (0, 165, 255)
    else: return "CENTERED", (0, 255, 0)

def get_ego_lane_poly(width, height):
    return np.array([[
        (int(width * 0.25), height),             
        (int(width * 0.75), height),             
        (int(width * 0.55), int(height * 0.55)), 
        (int(width * 0.45), int(height * 0.55))  
    ]], dtype=np.int32)

def download_youtube_video(url):
    print(f"[*] Downloading YouTube video locally to prevent disconnects...")
    safe_name = get_safe_filename(url)
    temp_filename = f"temp_dl_{safe_name}.mp4"
    
    if os.path.exists(temp_filename):
        return temp_filename
        
    ydl_opts = {
        'format': 'best[height<=720][ext=mp4]/best', 
        'outtmpl': temp_filename,
        'quiet': False,  # Shows you the download progress!
        'no_warnings': True
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return temp_filename
    except Exception as e:
        print(f"❌ Error downloading YouTube video: {e}")
        return None

# ==========================================
# BATCH RENDER & TRACKING SYSTEM
# ==========================================
def process_single_video(video_path, use_lane, model_general, model_car, show_preview):
    is_youtube = video_path.startswith("http")
    temp_dl_file = None
    
    if is_youtube:
        temp_dl_file = download_youtube_video(video_path)
        if not temp_dl_file: return
        cap = cv2.VideoCapture(temp_dl_file)
    else:
        if not os.path.exists(video_path):
            print(f"Skipping {video_path}: File not found.")
            return
        cap = cv2.VideoCapture(video_path)

    start_global_sec, target_end_sec, part_num, output_filename = audit_and_get_resume_state(video_path)

    start_global_sec, target_end_sec, part_num, output_filename = audit_and_get_resume_state(video_path)
    print(f"\n>>> Processing: {video_path}")
    print(f"    Resuming from: {start_global_sec}s | Saving to: {output_filename}")
    
    if start_global_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_global_sec * 1000)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30.0
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, int(fps), (w, h))

    last_left = None; last_right = None; alpha = 0.8
    track_history = {} 
    logged_tracks = set() 
    active_threats = {}     
    threat_cooldowns = {}   
    
    ego_lane_poly = get_ego_lane_poly(w, h)
    screen_center_x = w / 2
    frame_count = 0
    start_time = time.time()
    inference_size = 640 if model_car and "openvino" in MODEL_CAR_PATH.lower() else 480

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            local_current_sec = frame_count / fps 
            global_current_sec = start_global_sec + local_current_sec
            
            # File Audit check logic (Synced with Colab)
            if target_end_sec is not None and global_current_sec >= target_end_sec:
                print(f"✅ Repaired missing gap locally. Reached {target_end_sec:.1f}s.")
                break
                
            final_frame = frame.copy() 
            ego_drifting = False 
            scene_density = 0 

            # 1. LANE DETECTION
            if use_lane:
                try:
                    canny_img = canny(frame)
                    roi = region_of_interest(canny_img)
                    lines = cv2.HoughLinesP(roi, 2, np.pi/180, 50, np.array([]), minLineLength=20, maxLineGap=150)
                    l_fit, r_fit = average_slope_intercept(frame, lines)
                    if l_fit is not None: l_fit = tuple(alpha*np.array(l_fit)+(1-alpha)*np.array(last_left if last_left else l_fit)); last_left = l_fit
                    if r_fit is not None: r_fit = tuple(alpha*np.array(r_fit)+(1-alpha)*np.array(last_right if last_right else r_fit)); last_right = r_fit

                    left_line = make_coordinates(frame, l_fit)
                    right_line = make_coordinates(frame, r_fit)
                    line_img = np.zeros_like(frame)
                    if left_line is not None: cv2.line(line_img, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (255, 0, 0), 10)
                    if right_line is not None: cv2.line(line_img, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 0, 255), 10)
                    final_frame = cv2.addWeighted(final_frame, 0.8, line_img, 1, 1)
                    
                    status_text, status_color = get_lane_status(w, left_line, right_line)
                    if "DRIFTING" in status_text: ego_drifting = True
                    cv2.rectangle(final_frame, (40, 40), (450, 90), (0,0,0), -1) 
                    cv2.putText(final_frame, f"STATUS: {status_text}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                except Exception: pass
                
            cv2.polylines(final_frame, [ego_lane_poly], True, (255, 255, 0), 1)

            # 2. CAR PHYSICS & THREAT TRACKING
            alert_level = "NONE"
            current_frame_reds = set()

            if model_car:
                try:
                    res_car = model_car.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, imgsz=inference_size)
                    
                    if res_car[0].boxes.id is not None:
                        boxes = res_car[0].boxes.xyxy.cpu().numpy().tolist()
                        track_ids = res_car[0].boxes.id.int().cpu().numpy().tolist()
                        class_ids = res_car[0].boxes.cls.int().cpu().numpy().tolist()
                        names = model_car.names
                        
                        scene_density = len(track_ids)

                        for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                            x1, y1, x2, y2 = map(int, box)
                            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                            current_width = max(1, x2 - x1)
                            current_height = max(1, y2 - y1)
                            
                            obj_name = names[cls_id] if cls_id in names else 'unknown'
                            real_width = REAL_WIDTHS.get(obj_name, REAL_WIDTHS['default'])
                            current_distance = (FOCAL_CONSTANT * real_width) / current_width

                            if track_id not in track_history: track_history[track_id] = deque(maxlen=HISTORY_LENGTH)
                            track_history[track_id].append((cx, cy, current_width, current_distance, local_current_sec))
                            
                            current_status = "SAFE"
                            label_text = f"{current_distance:.1f}m"
                            
                            if len(track_history[track_id]) >= 10:
                                hist = list(track_history[track_id])
                                old_cx, old_cy, old_dist = sum(p[0] for p in hist[:3])/3.0, sum(p[1] for p in hist[:3])/3.0, sum(p[3] for p in hist[:3])/3.0
                                new_cx, new_cy, new_dist = sum(p[0] for p in hist[-3:])/3.0, sum(p[1] for p in hist[-3:])/3.0, sum(p[3] for p in hist[-3:])/3.0
                                old_t, new_t = hist[1][4], hist[-2][4]
                                
                                dt = new_t - old_t
                                if dt > 0:
                                    vx, vy = (new_cx - old_cx) / dt, (new_cy - old_cy) / dt
                                    approach_speed_ms = (old_dist - new_dist) / dt
                                    approach_speed_kmh = max(-250.0, min(approach_speed_ms * 3.6, 250.0))
                                    
                                    speed_sign = "+" if approach_speed_kmh > 0 else ""
                                    label_text = f"{new_dist:.1f}m | {speed_sign}{approach_speed_kmh:.0f}km/h"

                                    future_cx, future_cy = int(new_cx + (vx * 1.0)), int(new_cy + (vy * 1.0))
                                    moving_inward = abs(future_cx - screen_center_x) < abs(new_cx - screen_center_x)
                                    
                                    is_cut_in = False
                                    if moving_inward and not (vy < -0.5):
                                        if ((h * 0.30) < cy < (h * 0.80)) and (abs(vx) > abs(vy) * 1.5) and abs(vx) > (LATERAL_THREAT_SPEED * 20):
                                            is_cut_in = True
                                        elif abs(vx) > (LATERAL_THREAT_SPEED * 30):
                                            is_cut_in = True
                                    
                                    future_point = (future_cx, future_cy + int((y2-y1)/2)) 
                                    will_hit_ego_lane = cv2.pointPolygonTest(ego_lane_poly, future_point, False) >= 0
                                    
                                    if will_hit_ego_lane:
                                        ttc = new_dist / approach_speed_ms if approach_speed_ms > 0.5 else 999.0
                                        
                                        stage_1_alert = new_dist < 8.0
                                        stage_2_alert = (new_dist <= 60.0 and ttc < 2.5)
                                        
                                        if stage_1_alert or stage_2_alert or is_cut_in:
                                            current_status = "RED"
                                            alert_level = "RED"
                                            current_frame_reds.add(track_id)
                                            cv2.line(final_frame, (int(new_cx), int(new_cy)), (future_cx, future_cy), (0, 255, 255), 3)
                                            
                                            if track_id not in logged_tracks:
                                                if track_id not in active_threats:
                                                    active_threats[track_id] = {
                                                        'start_local_sec': float(local_current_sec),
                                                        'dist': float(round(new_dist, 2)),
                                                        'speed': float(round(approach_speed_kmh, 2)),
                                                        'vx': float(round(vx, 2)),
                                                        'vy': float(round(vy, 2)),
                                                        'ttc': float(round(ttc, 2)),
                                                        'obj_class': obj_name,          
                                                        'bbox_cx': float(round(new_cx, 2)),    
                                                        'bbox_cy': float(round(new_cy, 2)),   
                                                        'bbox_w': int(current_width),         
                                                        'bbox_h': int(current_height),
                                                        'scene_density': int(scene_density),  
                                                        'ego_drifting': bool(ego_drifting)    
                                                    }
                                                threat_cooldowns[track_id] = 15 

                            color = (0, 0, 255) if current_status == "RED" else (0, 255, 0)
                            cv2.rectangle(final_frame, (x1, y1), (x2, y2), color, 2 if current_status == "SAFE" else 3)
                            
                            if current_status == "RED" or (y2 > h * 0.6):
                                label = label_text + (" !!!" if current_status == "RED" else "")
                                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                cv2.rectangle(final_frame, (x1, y1 - 25), (x1 + tw, y1), color, -1)
                                cv2.putText(final_frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
                except Exception as e:
                    pass

            expired_threats =[]
            for tid in list(threat_cooldowns.keys()):
                if tid not in current_frame_reds:
                    threat_cooldowns[tid] -= 1
                    if threat_cooldowns[tid] <= 0:
                        expired_threats.append(tid)
                        
            for tid in expired_threats:
                event_data = active_threats[tid]
                event_data['end_local_sec'] = float(local_current_sec)
                event_data['output_file'] = output_filename
                event_data['track_id'] = int(tid)  
                event_data['video_url'] = video_path
                event_data['evaluated'] = False
                
                append_event_log(event_data)
                print(f"[*] Context Logged: Car ID {tid} | Density: {scene_density} | At: X={int(event_data['bbox_cx'])} Y={int(event_data['bbox_cy'])}")
                
                logged_tracks.add(tid)
                del active_threats[tid]
                del threat_cooldowns[tid]

            if alert_level == "RED":
                 cv2.putText(final_frame, "COLLISION IMMINENT", (int(w/2)-250, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 5)
                 cv2.rectangle(final_frame, (0,0), (w,h), (0,0,255), 10)

            out.write(final_frame)
            
            if show_preview:
                preview = cv2.resize(final_frame, (960, 540))
                cv2.imshow("Autopilot Studio V9 - Local Run", preview)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            else:
                if frame_count % 300 == 0: print(f"   Processed {frame_count} frames... ({global_current_sec:.1f}s)")
            
            if frame_count % 30 == 0:
                update_tracking(video_path, part_num, start_global_sec, global_current_sec, output_filename)
                
            frame_count += 1
            
        if target_end_sec is None:
            update_tracking(video_path, part_num, start_global_sec, global_current_sec, output_filename, status="completed")
            print(f"✅ Video Finished and Marked Completed.")
        
    except KeyboardInterrupt:
        print("\n⚠️ Process Interrupted by User. Saving checkpoint...")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Session Ended. Time taken: {int(time.time()-start_time)}s")

# ==========================================
# PASS 2: AI TRAINING EVALUATOR (COLAB PATH FIX APPLIED)
# ==========================================
# ==========================================
# PASS 2: AI TRAINING EVALUATOR (CHRONOLOGICAL FIX APPLIED)
# ==========================================
# ==========================================
# PASS 2: AI TRAINING EVALUATOR (CRASH BUFFER + MEDIA CONTROLS)
# ==========================================
# ==========================================
# PASS 2: AI TRAINING EVALUATOR (MANUAL CONTROL FIX)
# ==========================================
# ==========================================
# PASS 2: AI TARGET EVALUATOR (PURPLE TARGET HIGHLIGHT)
# ==========================================
# ==========================================
# PASS 2: AI TARGET EVALUATOR (AUTO-PAUSE FIX)
# ==========================================
# ==========================================
# PASS 2: AI EVALUATOR (DELAYED PREVIEW + REPLAY FEATURE)
# ==========================================
# ==========================================
# PASS 2: AI EVALUATOR (EXACT-FRAME PREVIEW FIX)
# ==========================================
# ==========================================
# PASS 2: AI EVALUATOR (SPEED-GRADING EDITION)
# ==========================================
# ==========================================
# PASS 2: AI EVALUATOR (SPOTLIGHT + FRAME-PERFECT SEEK)
# ==========================================
# ==========================================
# PASS 2: AI EVALUATOR (TRUE-FRAME SYNC + SPOTLIGHT + SPEED GRADING)
# ==========================================
# ==========================================
# PASS 2: AI EVALUATOR (TRUE-FRAME SYNC + SPOTLIGHT)
# ==========================================
# ==========================================
# PASS 2: AI EVALUATOR (FREEZE-PROOF GRADING)
# ==========================================
# ==========================================
# PASS 2: AI EVALUATOR (FFMPEG PRE-RENDER ENGINE + SPOTLIGHT)
# ==========================================
def evaluate_events_menu():
    print("\n--- 🔍 AI TARGET EVALUATION STUDIO (LOCAL) ---")
    
    # 1. VERIFY FFMPEG IS INSTALLED
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("❌ FFMPEG NOT FOUND! This new frame-perfect method requires FFmpeg.")
        print("Please install FFmpeg and add it to your System PATH, then restart.")
        return

    events = load_json(EVENTS_FILE,[])
    unevaluated = [e for e in events if not e.get('evaluated', False)]
    
    if not unevaluated:
        print("✅ No new events to evaluate. You're all caught up!")
        return
        
    groups = {}
    for e in unevaluated:
        vf = os.path.basename(e['output_file'])
        if vf not in groups: groups[vf] =[]
        groups[vf].append(e)

    print("Select a processed video to evaluate:")
    files = list(groups.keys())
    for i, f in enumerate(files):
        print(f"[{i+1}] {f} ({len(groups[f])} events)")
        
    print(f"[{len(files)+1}] Go Back")
    
    try:
        choice = int(input("Choice: ").strip()) - 1
        if choice == len(files): return
        selected_file = files[choice]
    except:
        return
        
    if not os.path.exists(selected_file):
        print(f"❌ Error: {selected_file} not found locally.")
        return

    selected_events = sorted(groups[selected_file], key=lambda x: x.get('start_local_sec', 0))
    
    # Get base video properties
    cap_check = cv2.VideoCapture(selected_file)
    fps = cap_check.get(cv2.CAP_PROP_FPS)
    orig_w = cap_check.get(cv2.CAP_PROP_FRAME_WIDTH)
    orig_h = cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if fps == 0 or np.isnan(fps): fps = 30.0
    cap_check.release()
    
    window_name = "AI Context Player"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    temp_video_file = "temp_event_slice.mp4"

    for idx, ev in enumerate(selected_events):
        print(f"\n--- Event {idx+1}/{len(selected_events)} | Car ID: {ev['track_id']} ---")
        
        target_sec = ev['start_local_sec']
        
        # EXACT 3 SECONDS BEFORE AND AFTER
        start_play_sec = max(0.0, target_sec - 3.0)
        end_play_sec = ev['end_local_sec'] + 3.0
        duration = end_play_sec - start_play_sec
        
        # Relative frame number inside the new short clip
        target_frame_num = int((target_sec - start_play_sec) * fps)
        
        # --- THE ULTIMATE FIX: FFMPEG PRE-RENDER ---
        print(f"  [FFmpeg] Rendering perfect {duration:.1f}s slice into memory...")
        cmd =[
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(start_play_sec),
            "-i", selected_file,
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "ultrafast", # Renders instantly
            temp_video_file
        ]
        subprocess.run(cmd, check=True)
        
        cap_play = cv2.VideoCapture(temp_video_file)
        
        current_frame_num = 0
        last_frame = None
        paused_at_target = False
        discarded = False
        
        scale_x = 1280 / orig_w
        scale_y = 720 / orig_h
        cx, cy = ev.get('bbox_cx', 0), ev.get('bbox_cy', 0)
        bw, bh = ev.get('bbox_w', 0), ev.get('bbox_h', 0)
        
        px1, py1 = max(0, int((cx - bw/2) * scale_x)), max(0, int((cy - bh/2) * scale_y))
        px2, py2 = min(1280, int((cx + bw/2) * scale_x)), min(720, int((cy + bh/2) * scale_y))
        
        # STAGE 1: WATCH THE PERFECTLY CUT CLIP
        while cap_play.isOpened():
            ret, frame = cap_play.read()
            if not ret: break # Clip automatically ends at duration limit!
            
            last_frame = frame.copy()
            current_frame_num += 1
            
            # --- SPOTLIGHT TRIGGER ---
            if current_frame_num == target_frame_num and not paused_at_target:
                preview = cv2.resize(frame, (1280, 720))
                darkened = cv2.addWeighted(preview, 0.2, np.zeros_like(preview), 0.8, 0)
                if py2 > py1 and px2 > px1:
                    darkened[py1:py2, px1:px2] = preview[py1:py2, px1:px2]
                    cv2.rectangle(darkened, (px1, py1), (px2, py2), (0, 255, 255), 4)
                
                cv2.rectangle(darkened, (10, 10), (1200, 70), (0,0,0), -1)
                cv2.putText(darkened, f"[PAUSED] EXACT TARGET IDENTIFIED (ID: {ev['track_id']})", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(darkened, "Press SPACE to watch outcome | '0' to Discard", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow(window_name, darkened)
                
                while True:
                    key = cv2.waitKey(1) & 0xFF
                    if key == 32: # SPACE
                        paused_at_target = True; break
                    elif key == ord('0'):
                        discarded = True; break
                if discarded: break
                continue

            # Standard Playback
            play_preview = cv2.resize(frame, (1280, 720))
            cv2.rectangle(play_preview, (10, 10), (1200, 70), (0,0,0), -1)
            msg = f"Approaching Event..." if current_frame_num < target_frame_num else f"Watching Outcome..."
            cv2.putText(play_preview, f"{msg} | ENTER to Grade | '0' to Discard", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(window_name, play_preview)
            
            key = cv2.waitKey(int(1000/fps)) & 0xFF
            if key == ord('0'): discarded = True; break
            elif key in [13, ord('q'), ord('Q')]: break # ENTER
            
        cap_play.release()
        
        # Clean up the temp file after watching
        if os.path.exists(temp_video_file):
            try: os.remove(temp_video_file)
            except: pass

        if discarded:
            ev['evaluated'] = True; save_json(EVENTS_FILE, events); continue
        if last_frame is None: continue

        # STAGE 2: FREEZE-PROOF GRADING
        grading_frame = cv2.resize(last_frame, (1280, 720))

        # --- Q1: SEVERITY ---
        q1_img = grading_frame.copy()
        cv2.rectangle(q1_img, (10, 10), (900, 180), (0,0,0), -1)
        cv2.putText(q1_img, f"Q1: Evaluate Severity (ID: {ev['track_id']})", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(q1_img, "[1] Actual Crash  [2] Close Call  [3] False Alarm  [0] DISCARD", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        severity = None
        while severity is None and not discarded:
            cv2.imshow(window_name, q1_img)
            key = cv2.waitKey(1) & 0xFF 
            if key == ord('1'): severity = 1
            elif key == ord('2'): severity = 2
            elif key == ord('3'): severity = 3
            elif key == ord('0'): discarded = True

        if discarded:
            ev['evaluated'] = True; save_json(EVENTS_FILE, events); continue

        if severity == 3:
            target_valid, start_timing, end_quality, reward_score = 1, 'c', 1, 0
        else:
            # --- Q2: TIMING ---
            q2_img = grading_frame.copy()
            cv2.rectangle(q2_img, (10, 10), (900, 160), (0,0,0), -1)
            cv2.putText(q2_img, "Q2: Warning Timing?", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(q2_img, "[a] Too Late  [b] Too Early  [c] Perfect Timing", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            start_timing = None
            while start_timing is None:
                cv2.imshow(window_name, q2_img)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('a'), ord('A')]: start_timing = 'a'
                elif key in [ord('b'), ord('B')]: start_timing = 'b'
                elif key in [ord('c'), ord('C')]: start_timing = 'c'

            # --- Q3: END QUALITY ---
            q3_img = grading_frame.copy()
            cv2.rectangle(q3_img, (10, 10), (900, 160), (0,0,0), -1)
            cv2.putText(q3_img, "Q3: Warning End Quality?", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(q3_img, "[1] On-time[2] Dropped Too Early", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            end_quality = None
            while end_quality is None:
                cv2.imshow(window_name, q3_img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('1'): end_quality = 1
                elif key == ord('2'): end_quality = 2

            target_valid = 1
            base_score = {1: 100, 2: 80}.get(severity, 0)
            p_start = {'c': 0, 'b': 10, 'a': 30}.get(start_timing, 0) 
            p_end = {1: 0, 2: 20}.get(end_quality, 0)
            reward_score = max(0, base_score - p_start - p_end)

        # SAVE DATA
        with open(CSV_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                ev['track_id'], ev.get('obj_class', 'unknown'), ev.get('bbox_cx', 0), 
                ev.get('bbox_cy', 0), ev.get('bbox_w', 0), ev.get('bbox_h', 0), 
                ev.get('scene_density', 0), ev.get('ego_drifting', False), 
                ev['dist'], ev['speed'], ev['vx'], ev['vy'], ev['ttc'], 
                target_valid, severity, start_timing, end_quality, reward_score
            ])
            
        print(f"[*] ML Saved: ID {ev['track_id']} | Score: {reward_score}")
        ev['evaluated'] = True
        save_json(EVENTS_FILE, events)

    cv2.destroyAllWindows()
    print("\n✅ Session Complete.")

def main():
    while True:
        print("\n=== AUTOPILOT STUDIO V9.9 Local (Colab-Sync Edition) ===")
        print(" [1] Render & Extract Videos (Use this if you want to render locally)")
        print(" [2] Train AI Model via Event Grading (Use this for Colab Evaluation)")
        print(" [3] Exit")
        
        choice = input("Select mode: ").strip()
        
        if choice == '1':
            use_lane = input("1. Enable Lane Detection? (y/n): ").lower().strip() == 'y'
            use_m2 = input("2. Enable Car Physics Model? (y/n): ").lower().strip() == 'y'
            show_preview = input("3. Show Rendering Window? (Can slow down speed) (y/n): ").lower().strip() == 'y'
            
            model_general, model_car = None, None
            print("\nLoading Models...")
            
            if use_m2: 
                try:
                    model_car = YOLO(MODEL_CAR_PATH, task='detect')
                    print(f"✅ Successfully loaded {MODEL_CAR_PATH}!")
                    if "openvino" in MODEL_CAR_PATH.lower():
                        print("✅ Engaged Intel OpenVINO Engine (Auto-Device Selected)")
                except Exception as e:
                    print(f"❌ Failed to load Car Physics model: {e}")

            videos =[]
            print("\n--- VIDEO QUEUE ---")
            while True:
                v_input = input(f"Input URL or Path #{len(videos)+1} (or 'q' to start): ").strip()
                if v_input.upper() in ["END", "FINISH", "Q", ""]: break
                
                # Split the input by commas OR spaces to handle multi-link pasting
                raw_entries = re.split(r'[,\s]+', v_input)
                
                for entry in raw_entries:
                    entry = entry.strip()
                    if not entry: continue
                    if entry.startswith("http") or os.path.exists(entry):
                        videos.append(entry)
                        print(f"  + Added: {entry[:50]}...")
            
            for video_path in videos: 
                process_single_video(video_path, use_lane, model_general, model_car, show_preview)
                
        elif choice == '2': 
            evaluate_events_menu()
            
        elif choice == '3': 
            sys.exit(0)

if __name__ == "__main__":
    main()