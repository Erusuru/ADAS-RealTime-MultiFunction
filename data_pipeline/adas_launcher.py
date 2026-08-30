#!/usr/bin/env python3
"""
ADAS & BeamNG Autopilot Comprehensive Batch Evaluation & ML Training Studio
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Features:
  1. Dynamically detects ALL scripts (aiovidout and beamng_pilot series).
  2. Interleaved dataset execution (Positive/Negative).
  3. Auto-patches BeamNG scripts to run offline math models on .mp4 files:
     - Positive videos run at 40 km/h simulated speed.
     - Negative videos run at 30 km/h simulated speed.
  4. Master performance leaderboard, precision, recall, and overall rankings.
  5. Mathematical Vectorial AI boundary optimization using positive case logs.
  6. Injected path-preserving name generator to prevent 0MB file truncation.
"""

import os
import sys
import json
import csv
import re
import shutil
import subprocess
import numpy as np

# Dataset directories
BASE_DATASET_DIR = r"C:\DISKD\ADAS BEST BEFORE OCR UPDATE\nexar_collision_prediction"
SUB_DIRS = ["test-public", "test-private", "train"]

# Output files
MASTER_EVAL_FILE = "adas_master_evaluation.json"
CSV_TRAINING_FILE = "crash_training_data_v2.csv"
OPTIMIZED_AI_MODEL = "optimized_collision_ai.json"

# ==========================================
# DYNAMIC SCRIPT DISCOVERY
# ==========================================
def get_all_scripts():
    """Finds all aiovidout and beamng python scripts in the root directory."""
    scripts = []
    for file in os.listdir('.'):
        if file.endswith('.py') and file not in ["adas_ai_studio.py", "launcher.py"]:
            if file.startswith("aiovidout") or file.startswith("allinone") or file.startswith("beamng"):
                scripts.append(file)
    # Sort to ensure consistent execution order
    return sorted(scripts)

# ==========================================
# FILE SYSTEM & TRAVERSAL UTILS
# ==========================================
def scan_dataset():
    """Scans subdirectories and returns interleaved positive and negative videos."""
    positives = []
    negatives = []
    
    search_paths = [BASE_DATASET_DIR] + [os.path.join(BASE_DATASET_DIR, d) for d in SUB_DIRS]
    
    for path in search_paths:
        if not os.path.exists(path):
            continue
        for root, dirs, files in os.walk(path):
            folder_name = os.path.basename(root).lower()
            if folder_name == "positive":
                for file in files:
                    if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        positives.append(os.path.abspath(os.path.join(root, file)))
            elif folder_name == "negative":
                for file in files:
                    if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        negatives.append(os.path.abspath(os.path.join(root, file)))

    # Interleave: 1 Pos, 1 Neg, 1 Pos, 1 Neg
    interleaved = []
    pos_idx, neg_idx = 0, 0
    while pos_idx < len(positives) or neg_idx < len(negatives):
        if pos_idx < len(positives):
            interleaved.append((positives[pos_idx], "positive"))
            pos_idx += 1
        if neg_idx < len(negatives):
            interleaved.append((negatives[neg_idx], "negative"))
            neg_idx += 1
            
    return interleaved

# ==========================================
# AUTO-PATCHING ENGINE (THE FIX)
# ==========================================
INJECTED_BEAMNG_RUNNER = """
# ==========================================================
# INJECTED OFFLINE RUNNER FOR BEAMNG SCRIPTS
# ==========================================================
import sys
import json
import csv
import inspect
import cv2
import numpy as np
import math
import time

def run_offline(video_path, speed_kmh, ground_truth, output_filename):
    speed_kmh = float(speed_kmh)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    pw = globals().get('PROCESS_W', 640)
    ph = globals().get('PROCESS_H', 360)
    skip_frames = globals().get('YOLO_SKIP', globals().get('AI_SKIP_FRAMES', 2))
    
    out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), int(fps), (pw, ph))
    
    events_log = []
    metrics_rows = []
    active_threat_events = {}
    threat_cooldowns = {}
    logged_tracks = set()
    frame_count = 0
    
    # Architecture Detection
    is_arch_b = 'YOLOTracker' in globals() and 'AEB' in globals()
    
    if is_arch_b:
        yolo_tracker = YOLOTracker()
        lane_det = LaneDetector()
        aeb = AEB()
        roadseg = RoadSegmentor() if 'RoadSegmentor' in globals() else None
    else:
        lane_assist = LaneAssist()
        vision_tracker = VisionTracker()
        model_path = globals().get('MODEL_PATH', 'selfdriving.pt')
        import os
        if not os.path.exists(model_path): model_path = 'yolov8n.pt'
        model = YOLO(model_path)
        roadseg = RoadSegmentor() if 'RoadSegmentor' in globals() else None

    while cap.isOpened():
        ret, frame_raw = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame_raw, (pw, ph))
        overlay = frame.copy()
        local_sec = frame_count / fps
        
        max_threat = 0
        current_frame_reds = set()
        scene_density = 0

        if is_arch_b:
            ann = None
            l_line, r_line, cte, hdg, lane_cx, roi_y = lane_det.process(frame, ann)
            if l_line is not None and r_line is not None:
                active_ego_poly = np.array([
                    [l_line[0], l_line[1] + roi_y], [r_line[0], r_line[1] + roi_y],
                    [r_line[2], r_line[3] + roi_y], [l_line[2], l_line[3] + roi_y],
                ], dtype=np.int32)
                cv2.fillPoly(overlay, [active_ego_poly], (0, 35, 0))
            
            if roadseg and getattr(roadseg, 'ok', False):
                roadseg.process(frame)
            
            dets = yolo_tracker.process(frame, speed_kmh / 3.6)
            scene_density = len(dets)
            cam_min_d = min((d["dist"] for d in dets), default=float("inf"))
            aeb_lv, aeb_brk, thr_scale, ttc = aeb.update(float("inf"), cam_min_d, speed_kmh / 3.6)
            
            if aeb_lv >= 2: max_threat = 2
            elif aeb_lv == 1: max_threat = 1
                
            for det in dets:
                x1, y1, x2, y2 = det["box"]
                tid = det["tid"]
                if aeb_lv >= 2 and det["dist"] == cam_min_d:
                    current_frame_reds.add(tid)
                    if tid not in active_threat_events:
                        active_threat_events[tid] = {
                            "start_local_sec": local_sec, "dist": det["dist"], "speed": speed_kmh,
                            "vx": det["vx"], "vy": det["vy"], "ttc": det["ttc"],
                            "obj_class": det["name"], "bbox_cx": det["cx"], "bbox_cy": det["cy"],
                            "bbox_w": x2-x1, "bbox_h": y2-y1, "scene_density": scene_density,
                            "ego_drifting": False, "output_file": output_filename, "track_id": tid,
                            "video_url": video_path, "target_correct": 1 if ground_truth=="positive" else 0,
                            "severity": 1 if ground_truth=="positive" else 3,
                            "reward_score": 100 if ground_truth=="positive" else 0
                        }
                        threat_cooldowns[tid] = 15
        else:
            dev, found_lanes, lane_poly = lane_assist.process(frame, overlay)
            active_ego_poly = lane_poly
            cv2.fillPoly(overlay, [active_ego_poly], (0, 30, 0))
            
            if roadseg and getattr(roadseg, 'ok', False):
                roadseg.process(frame)
                
            if frame_count % skip_frames == 0:
                results = model.track(frame, imgsz=640, verbose=False, conf=0.35, persist=True)
                if results[0].boxes and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().numpy()
                    class_ids = results[0].boxes.cls.int().cpu().numpy()
                    scene_density = len(track_ids)
                    
                    for box, tid, cid in zip(boxes, track_ids, class_ids):
                        sig = inspect.signature(vision_tracker.analyze)
                        if len(sig.parameters) == 8: 
                            data = vision_tracker.analyze(box, tid, cid, speed_kmh, pw, ph, active_ego_poly)
                        elif len(sig.parameters) == 7:
                            data = vision_tracker.analyze(box, tid, cid, speed_kmh, pw, ph, active_ego_poly)
                        elif len(sig.parameters) == 6:
                            data = vision_tracker.analyze(box, tid, cid, speed_kmh, active_ego_poly)
                        else:
                            data = vision_tracker.analyze(box, tid, cid, speed_kmh, active_ego_poly)
                            
                        if data:
                            if data["threat"] > max_threat: max_threat = data["threat"]
                            if data["threat"] == 2:
                                current_frame_reds.add(tid)
                                if tid not in active_threat_events:
                                    active_threat_events[tid] = {
                                        "start_local_sec": local_sec, "dist": data["dist"], "speed": speed_kmh,
                                        "vx": 0.0, "vy": 0.0, "ttc": data.get("ttc", data["dist"]/(speed_kmh/3.6+0.01)),
                                        "obj_class": "car", "bbox_cx": (box[0]+box[2])/2, "bbox_cy": (box[1]+box[3])/2,
                                        "bbox_w": box[2]-box[0], "bbox_h": box[3]-box[1], "scene_density": scene_density,
                                        "ego_drifting": False, "output_file": output_filename, "track_id": tid,
                                        "video_url": video_path, "target_correct": 1 if ground_truth=="positive" else 0,
                                        "severity": 1 if ground_truth=="positive" else 3,
                                        "reward_score": 100 if ground_truth=="positive" else 0
                                    }
                                    threat_cooldowns[tid] = 15

        expired = []
        for tid in list(threat_cooldowns.keys()):
            if tid not in current_frame_reds:
                threat_cooldowns[tid] -= 1
                if threat_cooldowns[tid] <= 0:
                    expired.append(tid)
        for tid in expired:
            ev = active_threat_events[tid]
            ev["end_local_sec"] = local_sec
            ev["evaluated"] = True
            events_log.append(ev)
            metrics_rows.append([
                ev["track_id"], ev["obj_class"], ev["bbox_cx"], ev["bbox_cy"],
                ev["bbox_w"], ev["bbox_h"], ev["scene_density"], ev["ego_drifting"],
                round(ev["dist"], 2), round(ev["speed"], 2), ev["vx"], ev["vy"],
                round(ev["ttc"], 2), ev["target_correct"], ev["severity"], "c", 1, ev["reward_score"]
            ])
            logged_tracks.add(tid)
            del active_threat_events[tid]
            del threat_cooldowns[tid]

        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        if max_threat == 2:
            cv2.rectangle(frame, (0, 0), (pw, ph), (0, 0, 255), 8)
            
        cv2.rectangle(frame, (0, 0), (240, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"SIM SPEED: {speed_kmh:.0f} km/h", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out.write(frame)
        frame_count += 1
        
    cap.release()
    out.release()
    
    for tid, ev in active_threat_events.items():
        ev["end_local_sec"] = frame_count / fps
        ev["evaluated"] = True
        events_log.append(ev)
        metrics_rows.append([
            ev["track_id"], ev["obj_class"], ev["bbox_cx"], ev["bbox_cy"],
            ev["bbox_w"], ev["bbox_h"], ev["scene_density"], ev["ego_drifting"],
            round(ev["dist"], 2), round(ev["speed"], 2), ev["vx"], ev["vy"],
            round(ev["ttc"], 2), ev["target_correct"], ev["severity"], "c", 1, ev["reward_score"]
        ])
        
    with open("events_log.json", "w") as f:
        json.dump(events_log, f, indent=4)
    with open("crash_training_data_v2.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Track_ID", "Obj_Class", "BBox_CX", "BBox_CY", "BBox_W", "BBox_H", 
            "Scene_Density", "Ego_Drifting", "Distance_m", "Approach_Speed_kmh", 
            "Lateral_Speed_vx", "Vertical_Speed_vy", "TTC_sec", "Target_Correct",
            "Severity", "Start_Timing", "End_Quality", "Reward_Score"
        ])
        writer.writerows(metrics_rows)

if __name__ == "__main__":
    run_offline(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
"""

def prepare_working_directory(folder_name, script_name):
    """Creates isolated folders, copies weights, and injects safe naming/offline runners."""
    os.makedirs(folder_name, exist_ok=True)
    dest_script = os.path.join(folder_name, script_name)
    try:
        shutil.copy2(script_name, dest_script)
    except Exception as e:
        print(f"[-] Error copying script: {e}")
        return False

    is_beamng = script_name.startswith("beamng")

    try:
        with open(dest_script, "r", encoding="utf-8") as f:
            code = f.read()
            
        # 1. Patch OpenVINO device migration crash
        if "model_car.to(" in code:
            code = code.replace("model_car.to(", "pass # model_car.to(")
            
        # 2. Append path-preserving unique file-name generator to overwrite base definitions
        override_naming_code = """
import os, re
def get_safe_filename(url_or_path):
    if not url_or_path or not isinstance(url_or_path, str): return "temp_video"
    if url_or_path.startswith("http"): return re.sub(r'[^a-zA-Z0-9]', '_', url_or_path)[:40]
    parts = os.path.normpath(url_or_path).split(os.sep)
    return re.sub(r'[^a-zA-Z0-9]', '_', "_".join(parts[-3:] if len(parts) >= 3 else parts))[:50]
"""
        if not is_beamng:
            code += override_naming_code
            
        # 3. If BeamNG script, strip live loop and inject offline runner
        if is_beamng:
            code = code.replace('if __name__ == "__main__":', 'if False:')
            code += INJECTED_BEAMNG_RUNNER

        with open(dest_script, "w", encoding="utf-8") as f:
            f.write(code)
    except Exception as e:
        print(f"[-] Auto-patch warning: {e}")

    # Link/Copy AI weight assets
    for item in os.listdir('.'):
        if item == folder_name or os.path.isdir(os.path.join(folder_name, item)):
            continue
        src = os.path.abspath(item)
        dst = os.path.abspath(os.path.join(folder_name, item))
        if item.endswith('.pt') or item.endswith('.yaml') or 'openvino' in item.lower():
            if os.path.exists(dst): continue
            try:
                if os.path.isdir(src): os.symlink(src, dst, target_is_directory=True)
                else: os.symlink(src, dst)
            except Exception:
                try:
                    if os.path.isdir(src): shutil.copytree(src, dst)
                    else: shutil.copy2(src, dst)
                except Exception: pass
    return True

# ==========================================
# LAUNCH OPTION 1: BATCH EXECUTION & EVALUATION
# ==========================================
def run_evaluation_pipeline():
    print("\n--- 🔍 ADAS & BEAMNG BATCH EXECUTION & EVALUATION PIPELINE ---")
    videos = scan_dataset()
    if not videos:
        print(f"[-] No test videos found inside dataset subdirectories under {BASE_DATASET_DIR}.")
        return

    scripts = get_all_scripts()
    print(f"[+] Found {len(videos)} videos and {len(scripts)} scripts. Running interleaved queue...")
    
    master_log = {}
    if os.path.exists(MASTER_EVAL_FILE):
        try:
            with open(MASTER_EVAL_FILE, "r") as f:
                master_log = json.load(f)
        except Exception:
            pass

    for idx, (video_path, ground_truth) in enumerate(videos):
        video_name = os.path.basename(video_path)
        normalized = os.path.normpath(video_path)
        parts = normalized.split(os.sep)
        unique_video_key = "_".join(parts[-3:]) if len(parts) >= 3 else video_name
        
        print(f"\n==================================================")
        print(f"[{idx+1}/{len(videos)}] Processing: {unique_video_key} (Ground Truth: {ground_truth.upper()})")
        print(f"==================================================")
        
        if unique_video_key not in master_log:
            master_log[unique_video_key] = {
                "video_path": video_path,
                "ground_truth": ground_truth,
                "evaluations": {}
            }

        for script in scripts:
            folder_name = os.path.splitext(script)[0]
            is_beamng = script.startswith("beamng")
            
            if script in master_log[unique_video_key]["evaluations"] and master_log[unique_video_key]["evaluations"][script].get("status") == "completed":
                print(f"[+] Skipping {script} (Already evaluated for this file).")
                continue
                
            log_path = os.path.join(folder_name, "events_log.json")
            csv_path = os.path.join(folder_name, "crash_training_data_v2.csv")
            for p in [log_path, csv_path]:
                if os.path.exists(p):
                    try: os.remove(p)
                    except Exception: pass
                    
            prepare_working_directory(folder_name, script)
            
            print(f"\n[+] Executing {script} against {unique_video_key}...")
            try:
                if is_beamng:
                    # BEAMNG INLINE RUNNER
                    speed_kmh = "40.0" if ground_truth == "positive" else "30.0"
                    out_name = f"processed_{unique_video_key}_v1_p1.mp4"
                    proc = subprocess.Popen(
                        [sys.executable, "-u", script, video_path, speed_kmh, ground_truth, out_name],
                        cwd=folder_name, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                    )
                    stdout, stderr = proc.communicate(timeout=600)
                    success = (proc.returncode == 0)
                else:
                    # STANDARD AIOVIDOUT RUNNER
                    inputs = f"1\ny\ny\nn\n{video_path}\nq\n3\n"
                    if "roadseg" in script: inputs = f"1\ny\ny\nn\nn\n{video_path}\nq\n4\n"
                    elif "3G" in script: inputs = f"1\n{video_path}\nq\n2\n"

                    proc = subprocess.Popen(
                        [sys.executable, "-u", script],
                        cwd=folder_name, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                    )
                    stdout, stderr = proc.communicate(input=inputs, timeout=600)
                    success = (proc.returncode == 0)
                    
            except subprocess.TimeoutExpired:
                print(f"[-] Execution Timeout reached for {script}.")
                proc.kill()
                continue
            except Exception as e:
                print(f"[-] Execution error: {e}")
                continue

            detections = []
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r") as f: detections = json.load(f)
                except Exception: pass
            
            alerts_triggered = len(detections) > 0
            if ground_truth == "positive":
                system_classification = "positive" if alerts_triggered else "negative"
                summary_note = "real event was positive, system detected positive" if alerts_triggered else "real was positive, system said negative (Missed Event)"
            else:
                system_classification = "positive" if alerts_triggered else "negative"
                summary_note = "real event was negative, system detected positive (False Alarm)" if alerts_triggered else "real was negative, system said negative"

            metrics_rows = []
            if os.path.exists(csv_path):
                try:
                    with open(csv_path, "r") as f:
                        reader = csv.reader(f)
                        header = next(reader)
                        for row in reader:
                            if len(row) >= len(header): metrics_rows.append(row)
                except Exception: pass

            master_log[unique_video_key]["evaluations"][script] = {
                "status": "completed",
                "system_classification": system_classification,
                "summary": summary_note,
                "notes": detections,
                "metrics_records": metrics_rows
            }
            
            with open(MASTER_EVAL_FILE, "w") as f:
                json.dump(master_log, f, indent=4)
                
            print(f"[*] Summary [{folder_name}]: {summary_note} (Alerts: {len(detections)})")

# ==========================================
# LAUNCH OPTION 2: DEEP STATISTICAL COMPARISON & LEADERBOARD
# ==========================================
def calculate_system_leaderboard():
    print("\n--- 🤖 ADAS PERFORMANCE METRICS & COMPARISON LEADERBOARD ---")
    if not os.path.exists(MASTER_EVAL_FILE):
        print(f"[-] Missing evaluation data. Run option [1] first to generate {MASTER_EVAL_FILE}.")
        return

    with open(MASTER_EVAL_FILE, "r") as f:
        master_log = json.load(f)

    if not master_log:
        print("[-] Master evaluation JSON is empty.")
        return

    stats = {}
    for video, data in master_log.items():
        ground_truth = data["ground_truth"]
        for script, eval_data in data["evaluations"].items():
            if script not in stats:
                stats[script] = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "total_reward": 0.0, "reward_count": 0}
            
            sys_cls = eval_data.get("system_classification")
            
            if ground_truth == "positive":
                if sys_cls == "positive": stats[script]["TP"] += 1
                else: stats[script]["FN"] += 1
            else:
                if sys_cls == "positive": stats[script]["FP"] += 1
                else: stats[script]["TN"] += 1

            for record in eval_data.get("metrics_records", []):
                try:
                    reward = float(record[-1])
                    stats[script]["total_reward"] += reward
                    stats[script]["reward_count"] += 1
                except: pass

    print("\n" + "="*75)
    print(f"{'ADAS SCRIPT VERSION':<30} | {'ACC%':<7} | {'SENS%':<7} | {'SPEC%':<7} | {'PREC%':<7} | {'AVG REWARD':<10}")
    print("="*75)

    leaderboard = []
    for script, counts in stats.items():
        tp, tn, fp, fn = counts["TP"], counts["TN"], counts["FP"], counts["FN"]
        total = tp + tn + fp + fn
        
        accuracy = ((tp + tn) / total * 100) if total > 0 else 0.0
        sensitivity = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        specificity = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0.0
        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        
        avg_reward = (counts["total_reward"] / counts["reward_count"]) if counts["reward_count"] > 0 else 0.0
        overall_score = (accuracy * 0.4) + (sensitivity * 0.3) + (specificity * 0.3)
        leaderboard.append((script, accuracy, sensitivity, specificity, precision, avg_reward, overall_score, counts))

        print(f"{script:<30} | {accuracy:>6.1f}% | {sensitivity:>6.1f}% | {specificity:>6.1f}% | {precision:>6.1f}% | {avg_reward:>10.2f}")

    if leaderboard:
        leaderboard.sort(key=lambda x: x[6], reverse=True)
        best_overall = leaderboard[0][0]
        most_pos_correct = max(leaderboard, key=lambda x: x[7]["TP"])
        most_neg_correct = max(leaderboard, key=lambda x: x[7]["TN"])

        print("="*75)
        print(f"\n🏆 METRIC COMPILATION ANALYSIS:")
        print(f"  • Best version for POSITIVE cases (Crash Alerts) : {most_pos_correct[0]} ({most_pos_correct[7]['TP']} hits)")
        print(f"  • Best version for NEGATIVE cases (False Alarm Isolation): {most_neg_correct[0]} ({most_neg_correct[7]['TN']} passes)")
        print(f"  • OVERALL CHAMPION ADAS MODEL                    : {best_overall}")
        print("="*75)

# ==========================================
# LAUNCH OPTION 3: VECTORIAL COLLISION WARNING AI TRAINING (ONLY POSITIVE CASES)
# ==========================================
def train_vectorial_collision_ai():
    print("\n--- 🧠 TRAINING VECTORIAL COLLISION WARNING AI (POSITIVE CASES ONLY) ---")
    
    dataset_rows = []
    if os.path.exists(MASTER_EVAL_FILE):
        try:
            with open(MASTER_EVAL_FILE, "r") as f:
                master_log = json.load(f)
            for video, v_data in master_log.items():
                for script, s_eval in v_data.get("evaluations", {}).items():
                    for record in s_eval.get("metrics_records", []):
                        if len(record) >= 18: dataset_rows.append(record)
        except Exception: pass

    if not dataset_rows:
        print("[-] Missing training features. Generate logs first by running Option [1].")
        return

    positive_cases = []
    for r in dataset_rows:
        try:
            target_correct = int(r[13])
            severity = int(r[14])
            if target_correct == 1 and severity in [1, 2]:
                positive_cases.append(r)
        except (ValueError, IndexError): continue

    if len(positive_cases) < 10:
        print(f"[-] Not enough positive cases logged (Count: {len(positive_cases)}). Need at least 10 logged instances to train.")
        return

    print(f"[+] Loaded {len(positive_cases)} verified positive events. Preparing dataset training vectors...")

    X_features = []
    y_rewards = []
    
    for r in positive_cases:
        try:
            dist = float(r[8])
            speed = float(r[9])
            vx = float(r[10])
            vy = float(r[11])
            ttc = float(r[12])
            reward = float(r[17])
            X_features.append([dist, speed, vx, vy, ttc])
            y_rewards.append(reward)
        except Exception: continue

    X = np.array(X_features)
    y = np.array(y_rewards)

    print("[*] Learning vectorial alert coefficients using regression optimization...")
    try:
        A = np.hstack([np.ones((X.shape[0], 1)), X])
        weights, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
        
        model_payload = {
            "bias": float(weights[0]),
            "weight_distance_m": float(weights[1]),
            "weight_approach_speed_kmh": float(weights[2]),
            "weight_lateral_vx": float(weights[3]),
            "weight_vertical_vy": float(weights[4]),
            "weight_ttc_sec": float(weights[5]),
            "optimized_cases_trained": len(X_features),
            "residual_error": float(np.mean(residuals) if residuals.size > 0 else 0.0)
        }

        with open(OPTIMIZED_AI_MODEL, "w") as f:
            json.dump(model_payload, f, indent=4)

        print("\n" + "="*60)
        print("🎉 SUCCESS: VECTORIAL COLLISION WARNING AI TRAINED")
        print("="*60)
        print(f"  • Bias (Constant Offset)  : {model_payload['bias']:.4f}")
        print(f"  • Distance (m) weight     : {model_payload['weight_distance_m']:.4f}")
        print(f"  • Speed (km/h) weight     : {model_payload['weight_approach_speed_kmh']:.4f}")
        print(f"  • Lateral Speed (vx) weight: {model_payload['weight_lateral_vx']:.4f}")
        print(f"  • Vertical Speed (vy) weight: {model_payload['weight_vertical_vy']:.4f}")
        print(f"  • TTC (sec) weight        : {model_payload['weight_ttc_sec']:.4f}")
        print(f"  • Output Model Saved To   : {OPTIMIZED_AI_MODEL}")
        print("="*60)
        print("Formula: Warning Risk = Bias + (w1 * Dist) + (w2 * Speed) + (w3 * vx) + (w4 * vy) + (w5 * TTC)")
        print("="*60)

    except Exception as e:
        print(f"[-] Training optimization failed: {e}")

# ==========================================
# MAIN DISPATCH LOOP
# ==========================================
def main():
    while True:
        print("\n=== ADAS & BEAMNG COMPREHENSIVE BATCH EVALUATION STUDIO ===")
        print(" [1] Run ADAS Evaluation Pipeline (All videos across ALL scripts)")
        print(" [2] Generate Statistical Performance Comparison Leaderboard")
        print(" [3] Train Vectorial Collision Warning AI (Only Positive Cases)")
        print(" [4] Exit")
        
        choice = input("Select operation: ").strip()
        if choice == '1':
            run_evaluation_pipeline()
        elif choice == '2':
            calculate_system_leaderboard()
        elif choice == '3':
            train_vectorial_collision_ai()
        elif choice == '4':
            sys.exit(0)

if __name__ == "__main__":
    main()