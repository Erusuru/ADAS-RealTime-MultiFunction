#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              ADAS BENCHMARK SUITE v2.0  —  DOĞRULUK & VECTOR ML             ║
║  Tüm scriptlerin konfigürasyonlarını tek çatı altında ardışık test eder.    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import time
import os
import sys
import csv
import json
import warnings
import datetime as _dt
import threading as _threading
from collections import deque
from typing import Optional, Tuple, Dict, Any, List

warnings.filterwarnings("ignore")

# ─── Klasör ve Dosya Ayarları ───────────────────────────────────────────────
BASE_DATASET_DIR = r"C:\DISKD\ADAS BEST BEFORE OCR UPDATE\nexar_collision_prediction"
LAUNCHER_SUB_DIRS = ["test-public", "test-private", "train"]
POS_FOLDER_NAMES = {"positive", "pos", "positives"}
NEG_FOLDER_NAMES = {"negative", "neg", "negatives"}

BENCHMARK_OUT_DIR   = "benchmark_out"
BENCHMARK_CSV       = os.path.join(BENCHMARK_OUT_DIR, "benchmark_results.csv")
LEADERBOARD_JSON    = os.path.join(BENCHMARK_OUT_DIR, "master_leaderboard.json")
VECTOR_TRAIN_CSV    = os.path.join(BENCHMARK_OUT_DIR, "vector_training_data.csv")
DETAILED_LOG_TXT    = os.path.join(BENCHMARK_OUT_DIR, "detailed_run_log.txt")

os.makedirs(BENCHMARK_OUT_DIR, exist_ok=True)

VECTOR_TRAIN_COLS = [
    "Config_ID", "Video_Name", "Ground_Truth", "Confusion",
    "System_Alert", "Dist_m", "App_Speed_kmh", "Vx", "Vy", "TTC_sec",
    "Total_Alerts", "First_Alert_Frame", "Max_Obj_Count",
    "Avg_FPS", "Total_Frames",
    "Night_Mode",
    "Focal", "Filter_Mode", "Alert_Mode", "YOLO_Conf",
]

if not os.path.exists(VECTOR_TRAIN_CSV):
    with open(VECTOR_TRAIN_CSV, mode='w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(VECTOR_TRAIN_COLS)

_LOG_LOCK = _threading.Lock()

def vlog(msg: str, to_console: bool = True) -> None:
    """Aşırı detaylı log: hem konsola hem de txt dosyasına yazar."""
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    line = f"[{ts}] {msg}"
    if to_console:
        print(line)
    try:
        with _LOG_LOCK:
            with open(DETAILED_LOG_TXT, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception: pass

# ─── Optional Kütüphaneler ──────────────────────────────────────────────────
try:
    import torch
    TORCH_OK = True
    CUDA_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
except ImportError:
    TORCH_OK = False; CUDA_DEVICE = None

try:
    from ultralytics import YOLO
except ImportError:
    print("[HATA] ultralytics bulunamadı!  pip install ultralytics")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 18 KONFIGÜRASYON
# ══════════════════════════════════════════════════════════════════════════════
METHOD_CONFIGS: List[Dict[str, Any]] = [
    {"id": 1, "name": "V7.5 Basic CPU", "source": "aiovidout.py", "focal_constant": 1200, "history_len": 10, "lateral_threat_speed": 2.0, "yolo_conf": 0.40, "yolo_imgsz": 640, "lane_alpha": 0.80, "ego_lane_mode": "static", "dist_filter": "raw", "alert_mode": "simple", "dist_red": 30.0, "half": False, "device": "cpu", "roadseg_config": None, "cliff_guard": False, "cut_in_bypass": False},
    {"id": 2, "name": "aiovidout2 Stage Alerts", "source": "aiovidout2.py", "focal_constant": 1200, "history_len": 15, "lateral_threat_speed": 2.0, "yolo_conf": 0.35, "yolo_imgsz": 480, "lane_alpha": 0.80, "ego_lane_mode": "static", "dist_filter": "raw", "alert_mode": "stage12", "stage1_dist": 8.0, "stage2_dist": 60.0, "ttc_warn": 2.5, "half": False, "device": "cpu", "roadseg_config": None, "cliff_guard": False, "cut_in_bypass": False},
    {"id": 3, "name": "aiovidout3 OpenVINO imgsz=640", "source": "aiovidout3.py", "focal_constant": 1200, "history_len": 15, "lateral_threat_speed": 2.0, "yolo_conf": 0.35, "yolo_imgsz": 640, "lane_alpha": 0.80, "ego_lane_mode": "static", "dist_filter": "raw", "alert_mode": "stage12", "stage1_dist": 8.0, "stage2_dist": 60.0, "ttc_warn": 2.5, "half": False, "device": "cpu", "roadseg_config": None, "cliff_guard": False, "cut_in_bypass": False},
    {"id": 4, "name": "aiovidout4 V9 conf=0.30", "source": "aiovidout4.py", "focal_constant": 1200, "history_len": 15, "lateral_threat_speed": 2.0, "yolo_conf": 0.30, "yolo_imgsz": 480, "lane_alpha": 0.80, "ego_lane_mode": "static", "dist_filter": "raw", "alert_mode": "stage12", "stage1_dist": 8.0, "stage2_dist": 60.0, "ttc_warn": 2.5, "half": False, "device": "cpu", "roadseg_config": None, "cliff_guard": False, "cut_in_bypass": False},
    {"id": 5, "name": "aiovidout5 V9 HW-Profile", "source": "aiovidout5.py", "focal_constant": 1200, "history_len": 15, "lateral_threat_speed": 2.0, "yolo_conf": 0.35, "yolo_imgsz": 480, "lane_alpha": 0.80, "ego_lane_mode": "static", "dist_filter": "raw", "alert_mode": "stage12", "stage1_dist": 8.0, "stage2_dist": 60.0, "ttc_warn": 2.5, "half": False, "device": "cpu", "roadseg_config": None, "cliff_guard": False, "cut_in_bypass": False},
    {"id": 6, "name": "aiovidout6 V9.5 DynamicEgo+Kalman", "source": "aiovidout6.py", "focal_constant": 1200, "history_len": 15, "lateral_threat_speed": 2.0, "yolo_conf": 0.35, "yolo_imgsz": 480, "lane_alpha": 0.80, "ego_lane_mode": "dynamic", "dist_filter": "kalman", "alert_mode": "stage12", "stage1_dist": 8.0, "stage2_dist": 60.0, "ttc_warn": 2.5, "half": False, "device": "cpu", "roadseg_config": None, "cliff_guard": False, "cut_in_bypass": True},
    {"id": 7, "name": "aiovidout7 V9.7 TwinLite(medium)+Kalman", "source": "aiovidout7_roadseg.py", "focal_constant": 1200, "history_len": 15, "lateral_threat_speed": 2.0, "yolo_conf": 0.35, "yolo_imgsz": 480, "lane_alpha": 0.80, "ego_lane_mode": "dynamic", "dist_filter": "kalman", "alert_mode": "stage12", "stage1_dist": 8.0, "stage2_dist": 60.0, "ttc_warn": 2.5, "half": False, "device": "cuda:0", "roadseg_config": "medium", "roadseg_repo_dir": r"C:\DISKD\ADAS BEST BEFORE OCR UPDATE\TwinLiteNetPlus", "roadseg_weights": r"C:\DISKD\ADAS BEST BEFORE OCR UPDATE\TwinLiteNetPlus\pretrained\medium.pth", "roadseg_imgsz": (640, 384), "roadseg_thresh": 0.5, "roadseg_every": 2, "cliff_guard": False, "cut_in_bypass": True},
    {"id": 8, "name": "aiovidout3G Colab T4 FP16", "source": "aiovidout3G.py", "focal_constant": 1200, "history_len": 15, "lateral_threat_speed": 2.0, "yolo_conf": 0.40, "yolo_imgsz": 640, "lane_alpha": 0.80, "ego_lane_mode": "static", "dist_filter": "raw", "alert_mode": "stage12", "stage1_dist": 8.0, "stage2_dist": 60.0, "ttc_warn": 2.5, "half": True, "device": "cuda:0", "roadseg_config": None, "cliff_guard": False, "cut_in_bypass": False},
    {"id": 9, "name": "BeamNG Pilot V1 Screen", "source": "beamngpilot.py", "focal_constant": 650, "history_len": 10, "lateral_threat_speed": 2.0, "yolo_conf": 0.30, "yolo_imgsz": 640, "lane_alpha": 0.80, "ego_lane_mode": "static", "dist_filter": "raw", "alert_mode": "simple", "dist_red": 15.0, "half": False, "device": "cpu", "roadseg_config": None, "cliff_guard": False, "cut_in_bypass": False},
    {"id": 10, "name": "BeamNG Pilot V2 Radar Overlay", "source": "beamngpilot2.py", "focal_constant": 650, "history_len": 10, "lateral_threat_speed": 2.0, "yolo_conf": 0.30, "yolo_imgsz": 640, "lane_alpha": 0.80, "ego_lane_mode": "static", "dist_filter": "raw", "alert_mode": "simple", "dist_red": 15.0, "half": False, "device": "cpu", "roadseg_config": None, "cliff_guard": False, "cut_in_bypass": False},
    {"id": 11, "name": "BeamNG Pilot V7 conf=0.35", "source": "beamng_pilot_v7.py", "focal_constant": 650, "history_len": 10, "lateral_threat_speed": 2.0, "yolo_conf": 0.35, "yolo_imgsz": 640, "lane_alpha": 0.80, "ego_lane_mode": "static", "dist_filter": "raw", "alert_mode": "simple", "dist_red": 15.0, "half": False, "device": "cpu", "roadseg_config": None, "cliff_guard": False, "cut_in_bypass": False},
    {"id": 12, "name": "BeamNG V3.Tech Camera Sensor", "source": "beamngpilot_v3_tech.py", "focal_constant": 650, "history_len": 10, "lateral_threat_speed": 2.0, "yolo_conf": 0.30, "yolo_imgsz": 640, "lane_alpha": 0.80, "ego_lane_mode": "static", "dist_filter": "raw", "alert_mode": "simple", "dist_red": 15.0, "half": False, "device": "cpu", "roadseg_config": None, "cliff_guard": False, "cut_in_bypass": False},
    {"id": 13, "name": "BeamNG Pilot3 LiDAR64+Stanley", "source": "beamngpilot3.py", "focal_constant": 650, "history_len": 10, "lateral_threat_speed": 2.0, "yolo_conf": 0.35, "yolo_imgsz": 640, "lane_alpha": 0.65, "ego_lane_mode": "dynamic", "dist_filter": "kalman", "alert_mode": "aeb3", "crit_dist": 5.0, "warn_dist": 20.0, "crit_ttc": 1.2, "warn_ttc": 3.0, "half": False, "device": "cuda:0", "roadseg_config": None, "cliff_guard": False, "cut_in_bypass": False},
    {"id": 14, "name": "BeamNG V4 LiDAR32+RateLimits", "source": "beamng_pilot_v4.py", "focal_constant": 650, "history_len": 10, "lateral_threat_speed": 2.0, "yolo_conf": 0.35, "yolo_imgsz": 640, "lane_alpha": 0.60, "ego_lane_mode": "dynamic", "dist_filter": "kalman", "alert_mode": "aeb3", "crit_dist": 6.0, "warn_dist": 25.0, "crit_ttc": 1.5, "warn_ttc": 3.5, "half": False, "device": "cuda:0", "roadseg_config": None, "cliff_guard": False, "cut_in_bypass": False},
    {"id": 15, "name": "BeamNG V5 TwinLite(medium)", "source": "beamng_pilot_v5.py", "focal_constant": 650, "history_len": 10, "lateral_threat_speed": 2.0, "yolo_conf": 0.35, "yolo_imgsz": 640, "lane_alpha": 0.60, "ego_lane_mode": "dynamic", "dist_filter": "kalman", "alert_mode": "aeb3", "crit_dist": 6.0, "warn_dist": 25.0, "crit_ttc": 1.5, "warn_ttc": 3.5, "half": False, "device": "cuda:0", "roadseg_config": "medium", "roadseg_repo_dir": r"C:\DISKD\ADAS BEST BEFORE OCR UPDATE\TwinLiteNetPlus", "roadseg_weights": r"C:\DISKD\ADAS BEST BEFORE OCR UPDATE\TwinLiteNetPlus\pretrained\medium.pth", "roadseg_imgsz": (640, 384), "roadseg_thresh": 0.5, "roadseg_every": 2, "cliff_guard": True, "cliff_foot_y": (0.80, 1.00), "cliff_foot_x": (0.32, 0.68), "cliff_min_frac": 0.35, "cliff_trigger_sec": 0.4, "cut_in_bypass": False},
    {"id": 16, "name": "BeamNG V6 AEB-Hysteresis", "source": "beamng_pilot_v6.py", "focal_constant": 650, "history_len": 10, "lateral_threat_speed": 2.0, "yolo_conf": 0.35, "yolo_imgsz": 640, "lane_alpha": 0.60, "ego_lane_mode": "dynamic", "dist_filter": "kalman", "alert_mode": "aeb3_hyst", "crit_dist": 6.0, "warn_dist": 25.0, "crit_ttc": 1.5, "warn_ttc": 3.5, "half": False, "device": "cuda:0", "roadseg_config": "medium", "roadseg_repo_dir": r"C:\DISKD\ADAS BEST BEFORE OCR UPDATE\TwinLiteNetPlus", "roadseg_weights": r"C:\DISKD\ADAS BEST BEFORE OCR UPDATE\TwinLiteNetPlus\pretrained\medium.pth", "roadseg_imgsz": (640, 384), "roadseg_thresh": 0.5, "roadseg_every": 2, "cliff_guard": True, "cliff_foot_y": (0.80, 1.00), "cliff_foot_x": (0.32, 0.68), "cliff_min_frac": 0.35, "cliff_trigger_sec": 0.4, "cut_in_bypass": False},
    {"id": 17, "name": "BeamNG V11 TwinLite(medium) 640x384", "source": "beamng_pilot_v11_roadseg.py", "focal_constant": 650, "history_len": 10, "lateral_threat_speed": 2.0, "yolo_conf": 0.35, "yolo_imgsz": 640, "lane_alpha": 0.80, "ego_lane_mode": "static", "dist_filter": "raw", "alert_mode": "simple", "dist_red": 15.0, "half": False, "device": "cuda:0", "roadseg_config": "medium", "roadseg_repo_dir": r"C:\DISKD\ADAS BEST BEFORE OCR UPDATE\TwinLiteNetPlus", "roadseg_weights": r"C:\DISKD\ADAS BEST BEFORE OCR UPDATE\TwinLiteNetPlus\pretrained\medium.pth", "roadseg_imgsz": (640, 384), "roadseg_thresh": 0.50, "roadseg_every": 2, "cliff_guard": False, "cut_in_bypass": False},
    {"id": 18, "name": "V9.5 Colab FP16 + EMA Dist", "source": "aiovidout3G.py", "focal_constant": 1200, "history_len": 15, "lateral_threat_speed": 2.0, "yolo_conf": 0.40, "yolo_imgsz": 640, "lane_alpha": 0.80, "ego_lane_mode": "static", "dist_filter": "ema", "dist_ema_alpha": 0.75, "alert_mode": "stage12", "stage1_dist": 8.0, "stage2_dist": 60.0, "ttc_warn": 2.5, "half": True, "device": "cuda:0", "roadseg_config": None, "cliff_guard": False, "cut_in_bypass": False},
]

REAL_WIDTHS = {"car": 2.0, "truck": 2.5, "bus": 3.0, "default": 2.0, 0: 2.0, 1: 2.5, 2: 3.0}

# ══════════════════════════════════════════════════════════════════════════════
# NUMPY KALMAN
# ══════════════════════════════════════════════════════════════════════════════
class TrackKalman:
    def __init__(self, cx: float, cy: float, d: float, dt: float = 1 / 30.0):
        self.dt = dt
        self.x = np.array([[cx], [cy], [d], [0.], [0.], [0.]], dtype=np.float64)
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 3] = self.F[1, 4] = self.F[2, 5] = dt
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0
        Q = np.eye(6, dtype=np.float64) * 1e-2
        Q[3:, 3:] *= 10.0
        self.Q = Q
        self.R = np.diag([0.5, 0.5, 5.0]).astype(np.float64)
        self.P = np.eye(6, dtype=np.float64) * 10.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, cx: float, cy: float, d: float):
        z = np.array([[cx], [cy], [d]], dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def state(self) -> Tuple[float, float, float, float, float, float]:
        return tuple(float(v) for v in self.x.flatten())

# ══════════════════════════════════════════════════════════════════════════════
# ŞERİT TESPİTİ
# ══════════════════════════════════════════════════════════════════════════════
def _adjust_gamma(img, gamma=0.7):
    inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table)

def _enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def _filter_colors(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    wm = cv2.inRange(hls, np.array([0, 140, 0]), np.array([255, 255, 255]))
    ym = cv2.inRange(hls, np.array([10, 50, 90]), np.array([40, 255, 255]))
    mask = cv2.bitwise_or(wm, ym)
    return cv2.bitwise_and(img, img, mask=mask)

def _canny_pipeline(img):
    darkened  = _adjust_gamma(img, 0.7)
    contrasted = _enhance_contrast(darkened)
    filtered   = _filter_colors(contrasted)
    gray       = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    blur       = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 40, 120)

def _roi_mask(edge_img, h, w):
    poly = np.array([[(int(w * 0.10), h), (int(w * 0.90), h),
                       (int(w * 0.55), int(h * 0.65)),
                       (int(w * 0.45), int(h * 0.65))]])
    mask = np.zeros_like(edge_img)
    cv2.fillPoly(mask, poly, 255)
    return cv2.bitwise_and(edge_img, mask)

def _avg_slope_intercept(frame, lines):
    left_fit, right_fit = [], []
    if lines is None: return None, None
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2 == x1: continue
        p = np.polyfit((x1, x2), (y1, y2), 1)
        if p[0] > 0.4:
            left_fit.append(p)
        elif p[0] < -0.4:
            right_fit.append(p)
    l = np.average(left_fit, axis=0) if left_fit else None
    r = np.average(right_fit, axis=0) if right_fit else None
    return l, r

def _make_line(frame, params):
    if params is None: return None
    slope, intercept = params
    if abs(slope) < 1e-4: return None
    y1 = frame.shape[0]; y2 = int(y1 * 0.65)
    try:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])
    except Exception: return None

def detect_lanes(frame, prev_left, prev_right, alpha: float):
    try:
        h, w = frame.shape[:2]
        edges = _canny_pipeline(frame)
        roi   = _roi_mask(edges, h, w)
        lines = cv2.HoughLinesP(roi, 2, np.pi / 180, 50, np.array([]), minLineLength=20, maxLineGap=150)
        l_fit, r_fit = _avg_slope_intercept(frame, lines)
        def ema(new_v, prev_v):
            if new_v is None: return prev_v
            if prev_v is None: return new_v
            return tuple(alpha * np.array(new_v) + (1 - alpha) * np.array(prev_v))
        l_fit = ema(l_fit, prev_left)
        r_fit = ema(r_fit, prev_right)
        return l_fit, r_fit, _make_line(frame, l_fit), _make_line(frame, r_fit)
    except Exception: return prev_left, prev_right, None, None

def static_ego_poly(w: int, h: int) -> np.ndarray:
    return np.array([[(int(w * 0.25), h), (int(w * 0.75), h), (int(w * 0.55), int(h * 0.55)), (int(w * 0.45), int(h * 0.55))]], dtype=np.int32)

def dynamic_ego_poly(w: int, h: int, left_line, right_line) -> np.ndarray:
    if left_line is None or right_line is None: return static_ego_poly(w, h)
    pts = np.array([[left_line[0],  left_line[1]], [right_line[0], right_line[1]], [right_line[2], right_line[3]], [left_line[2],  left_line[3]]], dtype=np.int32)
    return pts.reshape((-1, 1, 2))

# ══════════════════════════════════════════════════════════════════════════════
# YOL SEGMENTASYONU
# ══════════════════════════════════════════════════════════════════════════════
class RoadSegmentor:
    def __init__(self, cfg: Dict[str, Any]):
        self.ok = False
        self.cfg = cfg
        self._frame = 0
        self._da = None
        self._ll = None
        rs_cfg = cfg.get("roadseg_config")
        if not rs_cfg or not TORCH_OK: return
        repo = cfg.get("roadseg_repo_dir", r"C:\DISKD\ADAS BEST BEFORE OCR UPDATE\TwinLiteNetPlus")
        w_path = cfg.get("roadseg_weights", r"C:\DISKD\ADAS BEST BEFORE OCR UPDATE\TwinLiteNetPlus\pretrained\medium.pth")
        try:
            if repo not in sys.path: sys.path.insert(0, repo)
            from model.model import TwinLiteNetPlus
            import argparse
            args = argparse.Namespace()
            args.config = rs_cfg
            self.model = TwinLiteNetPlus(args).to(CUDA_DEVICE)
            ckpt = torch.load(w_path, map_location=CUDA_DEVICE, weights_only=False)
            state = ckpt.get("state_dict", ckpt)
            self.model.load_state_dict(state, strict=False)
            self.model.eval()
            self.imgsz = cfg.get("roadseg_imgsz", (640, 360))
            self.thresh = cfg.get("roadseg_thresh", 0.5)
            self.every = cfg.get("roadseg_every", 2)
            self.ok = True
        except Exception: pass

    def infer(self, frame_bgr: np.ndarray):
        self._frame += 1
        if not self.ok or (self._frame % self.every != 0): return self._da, self._ll
        try:
            rs = cv2.resize(frame_bgr, self.imgsz)
            t  = torch.from_numpy(rs).permute(2, 0, 1).float() / 255.0
            t  = t.unsqueeze(0).to(CUDA_DEVICE)
            with torch.no_grad(): da, ll = self.model(t)
            da_s = torch.softmax(da, dim=1)[:, 1]
            ll_s = torch.softmax(ll, dim=1)[:, 1]
            h, w  = frame_bgr.shape[:2]
            self._da = cv2.resize((da_s[0].cpu().numpy() > self.thresh).astype(np.uint8) * 255, (w, h))
            self._ll = cv2.resize((ll_s[0].cpu().numpy() > self.thresh).astype(np.uint8) * 255, (w, h))
        except Exception: pass
        return self._da, self._ll

    def overlay(self, frame: np.ndarray, da: Optional[np.ndarray], ll: Optional[np.ndarray]) -> np.ndarray:
        if da is not None:
            mask = np.zeros_like(frame)
            mask[da > 0] = (0, 100, 0)
            frame = cv2.addWeighted(frame, 1.0, mask, 0.35, 0)
        if ll is not None:
            mask = np.zeros_like(frame)
            mask[ll > 0] = (0, 0, 220)
            frame = cv2.addWeighted(frame, 1.0, mask, 0.45, 0)
        return frame

class CliffGuard:
    def __init__(self, cfg: Dict[str, Any]):
        self.foot_y = cfg.get("cliff_foot_y", (0.80, 1.00))
        self.foot_x = cfg.get("cliff_foot_x", (0.32, 0.68))
        self.min_frac = cfg.get("cliff_min_frac", 0.35)
        self.trig_sec = cfg.get("cliff_trigger_sec", 0.4)
        self._offroad_t = None

    def check(self, da_mask: Optional[np.ndarray], frame_h: int, frame_w: int, now: float) -> bool:
        if da_mask is None: return False
        y0, y1 = int(frame_h * self.foot_y[0]), int(frame_h * self.foot_y[1])
        x0, x1 = int(frame_w * self.foot_x[0]), int(frame_w * self.foot_x[1])
        roi = da_mask[y0:y1, x0:x1]
        frac = np.count_nonzero(roi) / max(roi.size, 1)
        if frac >= self.min_frac:
            self._offroad_t = None
            return False
        if self._offroad_t is None: self._offroad_t = now
        return (now - self._offroad_t) > self.trig_sec

# ══════════════════════════════════════════════════════════════════════════════
# DURUM VE GÜVENLİK SINIFLARI
# ══════════════════════════════════════════════════════════════════════════════
class MethodState:
    def __init__(self, cfg: Dict[str, Any], fps: float):
        self.cfg = cfg
        self.fps = fps
        self.track_hist = {}
        self.kalmans = {}
        self.ema_dists = {}
        self.prev_left = None
        self.prev_right = None
        self.brk_out = 0.0           
        self.offroad_t = None
        self.roadseg = RoadSegmentor(cfg)
        self.cliff = CliffGuard(cfg) if cfg.get("cliff_guard") else None
        self.alert_counts = {"CRIT": 0, "WARN": 0, "NONE": 0}
        self.total_alerts = 0
        self.detailed_events = []
        
        # Vektör AI eğitimi için sadeleştirilmiş telemetri
        self.highest_alert_issued = "NONE"
        self.max_threat_data = None
        # Genişletilmiş telemetri
        self.first_alert_frame: Optional[int] = None
        self.max_obj_count: int = 0
        self.brightness_samples: List[float] = []   # gece/gündüz tespiti için

def _get_distance(cfg, track_id, cx, cy, raw_dist, state: MethodState):
    mode = cfg.get("dist_filter", "raw")
    if mode == "raw": return raw_dist, cx, cy
    if mode == "ema":
        alpha = cfg.get("dist_ema_alpha", 0.75)
        fd = alpha * state.ema_dists.get(track_id, raw_dist) + (1 - alpha) * raw_dist
        state.ema_dists[track_id] = fd
        return fd, cx, cy
    if mode in ("kalman", "kalman_np"):
        dt = 1.0 / max(state.fps, 1.0)
        if track_id not in state.kalmans: state.kalmans[track_id] = TrackKalman(cx, cy, raw_dist, dt)
        kf = state.kalmans[track_id]
        kf.predict(); kf.update(cx, cy, raw_dist)
        sx, sy, sd, vx, vy, vd = kf.state()
        return sd, sx, sy
    return raw_dist, cx, cy

def _get_velocity(cfg, track_id, cx, cy, dist, now, state: MethodState):
    hist = state.track_hist.setdefault(track_id, deque(maxlen=cfg.get("history_len", 15)))
    hist.append((cx, cy, dist, now))
    if len(hist) < 3: return 0.0, 0.0, 0.0, 0.0
    ox, oy, od, ot = hist[0]
    nx, ny, nd, nt = hist[-1]
    dt = max(nt - ot, 1e-4)
    vx, vy, vd = (nx - ox) / dt, (ny - oy) / dt, (nd - od) / dt
    return vx, vy, max(-vd * 3.6, 0.0), max(-vd, 0.0)

def _check_alert(cfg, dist, ttc, is_cut_in, is_cut_in_bypass, will_hit_ego, state: MethodState) -> str:
    mode = cfg.get("alert_mode", "simple")
    if mode == "simple":
        return "RED" if (will_hit_ego and dist < cfg.get("dist_red", 30.0)) else "NONE"
    if mode == "stage12":
        if will_hit_ego or is_cut_in_bypass:
            if dist < cfg.get("stage1_dist", 8.0) or (dist <= cfg.get("stage2_dist", 60.0) and ttc < cfg.get("ttc_warn", 2.5)) or is_cut_in or is_cut_in_bypass:
                return "RED"
        return "NONE"
    if mode in ("aeb3", "aeb3_hyst"):
        cd, wd = cfg.get("crit_dist", 6.0), cfg.get("warn_dist", 25.0)
        ct, wt = cfg.get("crit_ttc", 1.5), cfg.get("warn_ttc", 3.5)
        if dist <= cd or ttc <= ct: return "CRIT"
        if dist <= wd * 0.45 or ttc <= wt * 0.5: return "WARN_STRONG"
        if dist <= wd or ttc <= wt: return "WARN"
    return "NONE"


# ══════════════════════════════════════════════════════════════════════════════
# ÇERÇEVE İŞLEMCİ (Sadeleştirilmiş Loglama İçerir)
# ══════════════════════════════════════════════════════════════════════════════
def process_frame(frame: np.ndarray, cfg: Dict[str, Any], state: MethodState, model: YOLO, now: float, frame_idx: int = -1) -> Tuple[np.ndarray, str]:
    h, w  = frame.shape[:2]
    final = frame.copy()

    # Gece/gündüz tespiti — her 30 frame'de bir örnekle (hızlı)
    if frame_idx % 30 == 0:
        try:
            gray_mean = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
            state.brightness_samples.append(gray_mean)
        except Exception: pass

    l_fit, r_fit, left_line, right_line = detect_lanes(frame, state.prev_left, state.prev_right, cfg.get("lane_alpha", 0.8))
    state.prev_left, state.prev_right = l_fit, r_fit

    line_img = np.zeros_like(final)
    if left_line is not None: cv2.line(line_img, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (255, 0, 0), 6)
    if right_line is not None: cv2.line(line_img, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 0, 255), 6)
    final = cv2.addWeighted(final, 0.85, line_img, 1.0, 0)

    ego_poly = dynamic_ego_poly(w, h, left_line, right_line) if cfg.get("ego_lane_mode") == "dynamic" else static_ego_poly(w, h)
    cv2.polylines(final, [ego_poly], True, (0, 255, 255), 1)

    da_mask = ll_mask = None
    if state.roadseg.ok:
        da_mask, ll_mask = state.roadseg.infer(frame)
        final = state.roadseg.overlay(final, da_mask, ll_mask)

    if state.cliff and da_mask is not None:
        if state.cliff.check(da_mask, h, w, now):
            cv2.putText(final, "⚠️ YOL DISI", (int(w * 0.1), int(h * 0.15)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.rectangle(final, (0, 0), (w, h), (0, 0, 255), 8)
            return final, "CLIFF"

    alert_level = "NONE"
    dev = cfg.get("device", "cpu")
    if dev == "auto": dev = "cuda:0" if (TORCH_OK and torch.cuda.is_available()) else "cpu"
    half = cfg.get("half", False) and TORCH_OK and torch.cuda.is_available()

    try:
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, imgsz=cfg.get("yolo_imgsz", 640), conf=cfg.get("yolo_conf", 0.35), device=dev, half=half)
    except Exception as e:
        cv2.putText(final, f"YOLO hatasi: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return final, "ERROR"

    if results[0].boxes.id is None: return final, "NONE"

    boxes = results[0].boxes.xyxy.cpu().numpy()
    track_ids = results[0].boxes.id.int().cpu().numpy()
    cls_ids = results[0].boxes.cls.int().cpu().numpy()
    names = model.names

    # Nesne sayısı takibi
    if len(boxes) > state.max_obj_count:
        state.max_obj_count = len(boxes)

    for box, tid, cid in zip(boxes, track_ids, cls_ids):
        x1, y1, x2, y2 = map(int, box)
        bw = max(x2 - x1, 1)
        cx_b, cy_b = int((x1 + x2) / 2), int((y1 + y2) / 2)

        obj_name = names.get(int(cid), "car") if isinstance(names, dict) else "car"
        real_w = REAL_WIDTHS.get(obj_name, REAL_WIDTHS.get(int(cid), 2.0))
        raw_dist = (cfg.get("focal_constant", 1200) * real_w) / bw

        filt_dist, fcx, fcy = _get_distance(cfg, tid, cx_b, cy_b, raw_dist, state)
        vx, vy, approach_kmh, approach_ms = _get_velocity(cfg, tid, fcx, fcy, filt_dist, now, state)

        fut_cx, fut_cy = int(fcx + vx * 1.0), int(fcy + vy * 1.0)
        fut_pt = (fut_cx, fut_cy + int((y2 - y1) / 2))
        will_hit = cv2.pointPolygonTest(ego_poly, fut_pt, False) >= 0

        moving_inward = abs(fut_cx - w/2.0) < abs(fcx - w/2.0)
        is_cut_in = moving_inward and (vy >= -0.5) and ((abs(vx) > abs(vy) * 1.5 and abs(vx) > cfg.get("lateral_threat_speed", 2.0) * 20 and (h * 0.30) < cy_b < (h * 0.80)) or abs(vx) > cfg.get("lateral_threat_speed", 2.0) * 30)
        is_cut_in_bypass = cfg.get("cut_in_bypass") and ((x1 < w * 0.15 and vx > 15.0) or (x2 > w * 0.85 and vx < -15.0))

        ttc = filt_dist / approach_ms if approach_ms > 0.5 else 999.0
        alv = _check_alert(cfg, filt_dist, ttc, is_cut_in, is_cut_in_bypass, will_hit, state)

        is_alert = (alv != "NONE")

        if is_alert:
            state.highest_alert_issued = alv
            if state.first_alert_frame is None:
                state.first_alert_frame = frame_idx
            if state.max_threat_data is None or alv in ("CRIT", "RED", "WARN_STRONG"):
                state.max_threat_data = [
                    cfg["id"], "VIDEO_NAME", "GROUND_TRUTH", alv, 
                    round(float(filt_dist), 2), round(float(approach_kmh), 2), 
                    round(float(vx), 2), round(float(vy), 2), round(float(ttc), 2)
                ]

        color = {"CRIT": (0, 0, 255), "WARN_STRONG": (0, 80, 255), "WARN": (0, 165, 255), "RED": (0, 0, 255)}.get(alv, (0, 255, 0))
        cv2.rectangle(final, (x1, y1), (x2, y2), color, 3 if is_alert else 2)

        label = f"ID{tid} {filt_dist:.1f}m"
        if approach_kmh > 1.0: label += f" {approach_kmh:.0f}k/h"
        if ttc < 999: label += f" TTC:{ttc:.1f}s"
        if is_alert: label += f" [{alv}]"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(final, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(final, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        if is_alert or is_cut_in:
            cv2.line(final, (cx_b, cy_b), (fut_cx, fut_cy), (0, 255, 255), 2)
            cv2.circle(final, fut_pt, 5, (0, 0, 255), -1)
            if alv not in (alert_level, "NONE"): alert_level = alv
            elif alert_level == "NONE": alert_level = alv

        if is_alert:
            state.detailed_events.append({
                "frame": int(frame_idx), "second": round(float(now), 3),
                "track_id": int(tid), "obj_class": str(obj_name), "alert_level": str(alv),
                "distance_m": round(float(filt_dist), 3), "speed_kmh": round(float(approach_kmh), 3),
                "ttc_sec": (round(float(ttc), 3) if ttc < 999 else None),
                "vx": round(float(vx), 4), "vy": round(float(vy), 4),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "is_cut_in": bool(is_cut_in), "is_cut_in_bypass": bool(is_cut_in_bypass), "will_hit_ego": bool(will_hit),
            })

    if alert_level in ("CRIT", "RED"):
        cv2.putText(final, "!!! CARPISMA !!!", (int(w / 2) - 150, int(h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4)
        cv2.rectangle(final, (0, 0), (w, h), (0, 0, 255), 12)
    elif alert_level in ("WARN_STRONG", "WARN"):
        cv2.putText(final, "UYARI", (int(w / 2) - 80, int(h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
        cv2.rectangle(final, (0, 0), (w, h), (0, 165, 255), 6)

    cv2.rectangle(final, (0, h - 36), (w, h), (20, 20, 20), -1)
    cv2.putText(final, f"C{cfg['id']:02d} | {cfg['name']} | focal={cfg['focal_constant']} conf={cfg['yolo_conf']} filter={cfg.get('dist_filter','raw')} alert={cfg['alert_mode']}", (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    if alert_level != "NONE": state.total_alerts += 1
    state.alert_counts["CRIT" if alert_level in ("CRIT", "RED") else "WARN" if "WARN" in alert_level else "NONE"] += 1

    return final, alert_level

# ══════════════════════════════════════════════════════════════════════════════
# BATCH VERİSETİ İŞLEME VE TEK VİDEO ÇALIŞTIRMA
# ══════════════════════════════════════════════════════════════════════════════
def classify_confusion(ground_truth: Optional[str], alarm_triggered: bool) -> str:
    if ground_truth not in ("positive", "negative"): return ""
    return "TP" if (ground_truth == "positive" and alarm_triggered) else "FN" if ground_truth == "positive" else "FP" if alarm_triggered else "TN"

def _safe_name(s: str) -> str:
    """Dosya sistemi için güvenli isim: boşluklar ve özel karakterler temizlenir."""
    return s.replace(" ", "_").replace("/", "-").replace("\\", "-").replace(":", "-").replace("*", "").replace("?", "").replace("\"", "").replace("<", "").replace(">", "").replace("|", "")

def run_config_on_video(video_path: str, cfg: Dict[str, Any], model: YOLO, out_dir: str, max_frames: int = 0, show_preview: bool = False, save_video: bool = True, ground_truth: Optional[str] = None, verbose: bool = True) -> Dict[str, Any]:
    # ── Tracker sıfırlama: bir önceki config'in ByteTracker state'i temizlenir ──
    try:
        if hasattr(model, 'predictor') and model.predictor is not None:
            model.predictor = None   # fresh predictor + fresh tracker her config için
    except Exception:
        pass

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return {"id": cfg["id"], "name": cfg["name"], "error": "video açilamadi"}

    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0: total = min(total, max_frames)

    # ── Her config için ayrı klasör: benchmark_out/cfg_01_V7.5_Basic_CPU/ ──
    cfg_folder_name = f"cfg_{cfg['id']:02d}_{_safe_name(cfg['name'])}"
    cfg_out_dir = os.path.join(out_dir, cfg_folder_name)
    os.makedirs(cfg_out_dir, exist_ok=True)

    # ── Benzersiz dosya adı: video adı + zaman damgası → üzerine yazılmaz ──
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = f"c{cfg['id']:02d}_{_safe_name(video_stem)}_{timestamp}.mp4"
    out_path = os.path.join(cfg_out_dir, out_filename)

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (w, h)) if save_video else None

    state = MethodState(cfg, fps)
    frame_idx = 0; t0 = time.time(); frame_times = []

    while cap.isOpened():
        if max_frames > 0 and frame_idx >= max_frames: break
        ret, frame = cap.read()
        if not ret: break

        ft0 = time.perf_counter()
        out_frame, alv = process_frame(frame, cfg, state, model, frame_idx / fps, frame_idx)
        frame_times.append(time.perf_counter() - ft0)

        if writer: writer.write(out_frame)
        if show_preview:
            win_title = f"C{cfg['id']:02d}/{cfg['name']} | {os.path.basename(video_path)}"
            cv2.imshow(win_title, cv2.resize(out_frame, (960, 540)))
            if cv2.waitKey(1) & 0xFF == ord("q"): break

        frame_idx += 1
        if verbose and frame_idx % 60 == 0:
            print(f"        [{(frame_idx / max(total, 1) * 100):5.1f}%] {frame_idx}/{total} kare  ~{(frame_idx / max(time.time() - t0, 0.001)):.1f} fps", end="\r")

    cap.release()
    if writer: writer.release()

    alarm_triggered = state.total_alerts > 0
    confusion = classify_confusion(ground_truth, alarm_triggered)

    # Gece/gündüz tespiti
    avg_brightness = float(np.mean(state.brightness_samples)) if state.brightness_samples else 128.0
    night_mode = 1 if avg_brightness < 60.0 else 0

    avg_fps  = round(frame_idx / max(time.time() - t0, 0.001), 1)

    # ── Vector CSV: her video için 1 satır yaz (pozitif VE negatif) ──────────
    if ground_truth is not None:
        # Tehdit anı verileri (yoksa sıfır — negatif/TN satırları da dahil)
        if state.max_threat_data is not None:
            state.max_threat_data[1] = os.path.basename(video_path)
            state.max_threat_data[2] = ground_truth.upper()
            threat_dist, threat_spd, threat_vx, threat_vy, threat_ttc = (
                state.max_threat_data[4], state.max_threat_data[5],
                state.max_threat_data[6], state.max_threat_data[7],
                state.max_threat_data[8],
            )
            sys_alert = state.max_threat_data[3]
        else:
            # Negatif video veya alarm üretmeyen config — sıfır doldur
            threat_dist = threat_spd = threat_vx = threat_vy = 0.0
            threat_ttc  = 999.0
            sys_alert   = "NONE"

        row = [
            cfg["id"],
            os.path.basename(video_path),
            ground_truth.upper(),
            confusion,
            sys_alert,
            threat_dist, threat_spd, threat_vx, threat_vy, threat_ttc,
            state.total_alerts,
            state.first_alert_frame if state.first_alert_frame is not None else -1,
            state.max_obj_count,
            avg_fps,
            frame_idx,
            night_mode,
            cfg.get("focal_constant", 0),
            cfg.get("dist_filter", "raw"),
            cfg.get("alert_mode", "simple"),
            cfg.get("yolo_conf", 0.35),
        ]
        with open(VECTOR_TRAIN_CSV, mode='a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(row)

    vlog(f"BİTTİ C{cfg['id']:02d} '{cfg['name']}' video='{os.path.basename(video_path)}' GT={ground_truth} CONF={confusion}", to_console=False)

    return {
        "id": cfg["id"], "name": cfg["name"], "source": cfg["source"], "frames": frame_idx, "total_alerts": state.total_alerts,
        "alerts_crit": state.alert_counts["CRIT"], "alerts_warn": state.alert_counts["WARN"],
        "avg_fps": round(frame_idx / max(time.time() - t0, 0.001), 1), "avg_lat_ms": round((sum(frame_times) / len(frame_times) * 1000) if frame_times else 0, 1),
        "roadseg": cfg.get("roadseg_config") or "-", "cliff": "✓" if cfg.get("cliff_guard") else "-",
        "output": out_path if save_video else None, "error": None, "ground_truth": ground_truth,
        "alarm_triggered": alarm_triggered, "confusion": confusion, "detailed_events": state.detailed_events,
    }

def _scan_videos_from_launcher() -> list:
    positives, negatives = [], []
    exts = ('.mp4', '.avi', '.mov', '.mkv')
    search_paths = [BASE_DATASET_DIR] + [os.path.join(BASE_DATASET_DIR, d) for d in LAUNCHER_SUB_DIRS]
    for base in search_paths:
        if not os.path.exists(base): continue
        for root, _, files in os.walk(base):
            folder_name = os.path.basename(root).lower()
            if folder_name in POS_FOLDER_NAMES:
                positives.extend([(os.path.join(root, f), "positive") for f in files if f.lower().endswith(exts)])
            elif folder_name in NEG_FOLDER_NAMES:
                negatives.extend([(os.path.join(root, f), "negative") for f in files if f.lower().endswith(exts)])

    interleaved = []
    p_idx, n_idx = 0, 0
    while p_idx < len(positives) or n_idx < len(negatives):
        if p_idx < len(positives): interleaved.append(positives[p_idx]); p_idx += 1
        if n_idx < len(negatives): interleaved.append(negatives[n_idx]); n_idx += 1
    return interleaved

def load_leaderboard():
    if os.path.exists(LEADERBOARD_JSON):
        try:
            with open(LEADERBOARD_JSON, "r") as f: return json.load(f)
        except Exception: pass
    lb = {}
    for cfg in METHOD_CONFIGS: lb[str(cfg["id"])] = {"name": cfg["name"], "TP": 0, "FP": 0, "TN": 0, "FN": 0, "processed_videos": []}
    return lb

def save_leaderboard(lb):
    with open(LEADERBOARD_JSON, "w") as f: json.dump(lb, f, indent=4)

def mode_benchmark_batch():
    print("\n" + "═" * 80)
    print("  [MODE 1] OTO-BATCH BENCHMARK (HER VİDEO İÇİN TÜM CONFIGLER ARDIŞIK)")
    print("═" * 80)

    videos = _scan_videos_from_launcher()
    if not videos:
        print(f"[HATA] {BASE_DATASET_DIR} altında video bulunamadı.")
        return

    print(f"\n🔍 {len(videos)} adet video bulundu (Pozitif ve Negatif harmanlanmış).")
    pt_files = [f for f in os.listdir(".") if f.endswith(".pt")]
    if pt_files: print(f"🤖 Bulunan model dosyaları: {', '.join(pt_files)}")
    model_path = input(f"🤖 YOLO model dosyası [selfdriving.pt]: ").strip() or "selfdriving.pt"
    if not os.path.exists(model_path):
        print(f"[HATA] Model bulunamadı: {model_path}"); return

    print("\n📋 18 konfigürasyon mevcut. Hepsini çalıştırmak için Enter,")
    sel_raw = input("   veya virgülle ayrılmış ID girin (ör: 1,2,6): ").strip()
    configs_to_run = [c for c in METHOD_CONFIGS if c["id"] in set([int(x) for x in sel_raw.replace(" ", "").split(",")])] if sel_raw else METHOD_CONFIGS

    show_preview = input("👁  İşlem sırasında videoları izle? (e/h) [h]: ").strip().lower() == "e"
    save_mp4 = input("💾  Çıktı videoları MP4 olarak kaydedilsin mi? (e/h) [e]: ").strip().lower() == "e"
    if save_mp4:
        print(f"📁  Videolar kaydedilecek: {BENCHMARK_OUT_DIR}/cfg_XX_<isim>/<video>_<tarih>.mp4")

    # MAX FRAMES KALDIRILDI: Videonun kaza anını yakalamak için sonuna kadar izlenmesi şarttır (0)
    max_frames = 0
    print(f"\n✅ Ayarlar: Tüm video işlenecek (kaza anını yakalamak için kesinti yok).")

    print(f"🔄 Model yükleniyor: {model_path}...")
    model = YOLO(model_path)
    leaderboard = load_leaderboard()

    # DÖNGÜ MANTIĞI: Önce tek bir video açılır, seçilen tüm configler o videoda sırayla koşturulur.
    for v_idx, (video_path, ground_truth) in enumerate(videos, 1):
        v_name = os.path.basename(video_path)
        print(f"\n▶ [{v_idx}/{len(videos)}] Video: {v_name} (Gerçek: {ground_truth.upper()})")
        
        for cfg in configs_to_run:
            cid = str(cfg["id"])
            if v_name in leaderboard[cid]["processed_videos"]:
                continue # Bu videoyu bu config ile daha önce işlemişiz, atla
                
            print(f"\n  {'─'*70}")
            print(f"  🚀 [{configs_to_run.index(cfg)+1}/{len(configs_to_run)}] C{cfg['id']:02d} | {cfg['name']}")
            print(f"     Video: {v_name}  ({v_idx}/{len(videos)})")
            print(f"  {'─'*70}")
            try:
                res = run_config_on_video(video_path, cfg, model, BENCHMARK_OUT_DIR, max_frames, show_preview, save_video=save_mp4, ground_truth=ground_truth, verbose=False)
                
                confusion = res.get("confusion", "")
                if confusion == "TP": leaderboard[cid]["TP"] += 1; res_str = "DOĞRU (Fren Yaptı)"
                elif confusion == "FN": leaderboard[cid]["FN"] += 1; res_str = "KAÇIRDI (Çarptı)"
                elif confusion == "TN": leaderboard[cid]["TN"] += 1; res_str = "DOĞRU (Pas Geçti)"
                elif confusion == "FP": leaderboard[cid]["FP"] += 1; res_str = "HATALI ALARM (Gereksiz Fren)"
                else: res_str = "BİLİNMİYOR"

                leaderboard[cid]["processed_videos"].append(v_name)
                save_leaderboard(leaderboard)
                print(f"    └─ Sonuç: {res_str}")
                if save_mp4 and res.get("output"):
                    print(f"    💾 Kaydedildi: {res['output']}")
                
            except Exception as e:
                print(f"    ❌ C{cfg['id']:02d} HATA: {e}")
            finally:
                cv2.destroyAllWindows()

    cv2.destroyAllWindows()
    print("\n✅ Tüm testler tamamlandı! Sonuçları görmek için [2]'yi seçin.")

def mode_comparison_leaderboard():
    print("\n" + "═" * 100)
    print("  [MODE 2] ADAS PERFORMANS LİDERLİK TABLOSU (ACCURACY ODAKLI)")
    print("═" * 100)

    if not os.path.exists(LEADERBOARD_JSON):
        print("[-] Veri yok. Önce [1] numaralı seçenekle test yapın.")
        return

    with open(LEADERBOARD_JSON, "r") as f: leaderboard = json.load(f)
        
    results = []
    for cid, data in leaderboard.items():
        tp, tn, fp, fn = data["TP"], data["TN"], data["FP"], data["FN"]
        total = tp + tn + fp + fn
        if total == 0: continue
        
        acc = (tp + tn) / total * 100.0
        recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        far = (fp / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0
        
        results.append({
            "id": int(cid), "name": data["name"], "total": total,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "Acc": acc, "Recall": recall, "FAR": far
        })

    results.sort(key=lambda x: (x["Acc"], x["Recall"]), reverse=True)

    print(f"{'ID':<3} | {'ALGORİTMA':<32} | {'VİD':<4} | {'DOĞRULUK(ACC)':<14} | {'HASSASİYET(TP)':<14} | {'HATALI ALARM(FP)':<14}")
    print("─" * 100)
    for r in results:
        print(f"{r['id']:<3} | {r['name']:<32} | {r['total']:<4} | %{r['Acc']:<13.1f} | %{r['Recall']:<13.1f} | %{r['FAR']:<13.1f} ({r['FP']} Kez)")
    print("─" * 100)

def mode_benchmark_top_performers():
    """
    Mod 3: En Kötü 5 Konfigürasyonu Atlayarak Hızlandırılmış Benchmark.
    Elenenler (ACC düşük veya FP %100):
      - ID 2  (aiovidout2 Stage Alerts)    → gündüz %60 ACC, %65 FP
      - ID 4  (aiovidout4 V9 conf=0.30)    → gündüz %60 ACC, %65 FP
      - ID 5  (aiovidout5 V9 HW-Profile)   → gündüz %60 ACC, %65 FP
      - ID 13 (BeamNG Pilot3 LiDAR64+Stanley) → %100 FP (23 Kez)
      - ID 14 (BeamNG V4 LiDAR32+RateLimits)  → %100 FP (23 Kez)
    Kalan 13 konfigürasyon koşturulur.
    """
    ELIMINATED_IDS = {2, 4, 5, 13, 14}
    top_configs = [c for c in METHOD_CONFIGS if c["id"] not in ELIMINATED_IDS]

    print("\n" + "═" * 80)
    print("  [MOD 3] 🏆 TOP PERFORMERS BENCHMARK (13 KONFİGÜRASYON — 5 KÖTÜsü ELENDİ)")
    print("═" * 80)
    print(f"  ❌ Elenen ID'ler: {sorted(ELIMINATED_IDS)}  (Düşük ACC / %100 Yanlış Alarm)")
    print(f"  ✅ Çalışacak ID'ler: {[c['id'] for c in top_configs]}")
    print("═" * 80)

    videos = _scan_videos_from_launcher()
    if not videos:
        print(f"[HATA] {BASE_DATASET_DIR} altında video bulunamadı.")
        return

    print(f"\n🔍 {len(videos)} adet video bulundu (Pozitif ve Negatif harmanlanmış).")
    pt_files = [f for f in os.listdir(".") if f.endswith(".pt")]
    if pt_files: print(f"🤖 Bulunan model dosyaları: {', '.join(pt_files)}")
    model_path = input(f"🤖 YOLO model dosyası [selfdriving.pt]: ").strip() or "selfdriving.pt"
    if not os.path.exists(model_path):
        print(f"[HATA] Model bulunamadı: {model_path}"); return

    show_preview = input("👁  İşlem sırasında videoları izle? (e/h) [h]: ").strip().lower() == "e"
    save_mp4 = input("💾  Çıktı videoları MP4 olarak kaydedilsin mi? (e/h) [h]: ").strip().lower() == "e"
    if save_mp4:
        print(f"📁  Videolar kaydedilecek: {BENCHMARK_OUT_DIR}/cfg_XX_<isim>/<video>_<tarih>.mp4")
    
    # MAX FRAMES KALDIRILDI: Videonun kaza anını yakalamak için sonuna kadar izlenmesi şarttır (0)
    max_frames = 0
    print(f"\n✅ Ayarlar: Tüm video işlenecek (kaza anını yakalamak için kesinti yok).")

    print(f"\n🔄 Model yükleniyor: {model_path}...")
    model = YOLO(model_path)
    leaderboard = load_leaderboard()

    # Elenen konfigürasyonlar için leaderboard'da yer tutucu oluştur (silme)
    for cid in ELIMINATED_IDS:
        if str(cid) not in leaderboard:
            leaderboard[str(cid)] = {"name": f"[ELENDİ] ID{cid}", "TP": 0, "FP": 0, "TN": 0, "FN": 0, "processed_videos": []}

    for v_idx, (video_path, ground_truth) in enumerate(videos, 1):
        v_name = os.path.basename(video_path)
        print(f"\n▶ [{v_idx}/{len(videos)}] Video: {v_name} (Gerçek: {ground_truth.upper()})")

        for cfg in top_configs:
            cid = str(cfg["id"])
            if v_name in leaderboard[cid]["processed_videos"]:
                continue

            print(f"\n  {'─'*70}")
            print(f"  🚀 [{top_configs.index(cfg)+1}/{len(top_configs)}] C{cfg['id']:02d} | {cfg['name']}")
            print(f"     Video: {v_name}  ({v_idx}/{len(videos)})")
            print(f"  {'─'*70}")
            try:
                res = run_config_on_video(video_path, cfg, model, BENCHMARK_OUT_DIR, max_frames, show_preview, save_video=save_mp4, ground_truth=ground_truth, verbose=False)

                confusion = res.get("confusion", "")
                if confusion == "TP": leaderboard[cid]["TP"] += 1; res_str = "DOĞRU (Fren Yaptı)"
                elif confusion == "FN": leaderboard[cid]["FN"] += 1; res_str = "KAÇIRDI (Çarptı)"
                elif confusion == "TN": leaderboard[cid]["TN"] += 1; res_str = "DOĞRU (Pas Geçti)"
                elif confusion == "FP": leaderboard[cid]["FP"] += 1; res_str = "HATALI ALARM (Gereksiz Fren)"
                else: res_str = "BİLİNMİYOR"

                leaderboard[cid]["processed_videos"].append(v_name)
                save_leaderboard(leaderboard)
                print(f"    └─ Sonuç: {res_str}")
                if save_mp4 and res.get("output"):
                    print(f"    💾 Kaydedildi: {res['output']}")

            except Exception as e:
                print(f"    ❌ C{cfg['id']:02d} HATA: {e}")
            finally:
                cv2.destroyAllWindows()

    cv2.destroyAllWindows()
    elim_str = ", ".join(f"ID{i}" for i in sorted(ELIMINATED_IDS))
    print(f"\n✅ Top Performers testi tamamlandı! ({elim_str} elendi, {len(top_configs)} config çalıştırıldı)")
    print("   Sonuçları görmek için [2]'yi seçin.")


def mode_fast_vector():
    """
    Mod 4: Maksimum hızda sadece vector_training_data.csv doldurmak için.
    - Video kaydı YOK (save_mp4=False)
    - Preview YOK (show_preview=False)
    - Sadece Top 13 config (elenenler hariç)
    - Varsayılan max_frames=150 (yaklaşık 5 saniye)
    - Bu modda kaldığın yerden devam edilir (processed_videos kontrolü)
    """
    ELIMINATED_IDS = {2, 4, 5, 13, 14}
    top_configs = [c for c in METHOD_CONFIGS if c["id"] not in ELIMINATED_IDS]

    print("\n" + "═" * 80)
    print("  [MOD 4] ⚡ HIZLI VEKTÖR MODU — Sadece CSV, No-Video, 150 Frame")
    print("═" * 80)
    print(f"  Config sayısı: {len(top_configs)}  |  Elenenler: {sorted(ELIMINATED_IDS)}")
    print(f"  Çıktı: {VECTOR_TRAIN_CSV}")
    print("═" * 80)

    videos = _scan_videos_from_launcher()
    if not videos:
        print(f"[HATA] {BASE_DATASET_DIR} altında video bulunamadı.")
        return

    print(f"\n🔍 {len(videos)} video bulundu.")
    pt_files = [f for f in os.listdir(".") if f.endswith(".pt")]
    if pt_files: print(f"🤖 Bulunan model dosyaları: {', '.join(pt_files)}")
    model_path = input(f"🤖 YOLO model dosyası [selfdriving.pt]: ").strip() or "selfdriving.pt"
    if not os.path.exists(model_path):
        print(f"[HATA] Model bulunamadı: {model_path}"); return

    max_fr_raw = input("⚡ Maksimum frame/video [150]: ").strip()
    max_frames = int(max_fr_raw) if max_fr_raw.isdigit() else 150
    print(f"\n✅ Ayarlar: max_frames={max_frames}, save_video=HAYIR, preview=HAYIR")
    print(f"🔄 Model yükleniyor: {model_path}...")

    model = YOLO(model_path)
    leaderboard = load_leaderboard()

    # Mevcut CSV'deki işlenmiş (video, config) çiftlerini oku — kaldığın yerden devam
    processed_pairs: set = set()
    if os.path.exists(VECTOR_TRAIN_CSV):
        try:
            with open(VECTOR_TRAIN_CSV, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        processed_pairs.add((row["Config_ID"], row["Video_Name"]))
                    except KeyError: pass
        except Exception: pass
    print(f"📂 Mevcut CSV'de {len(processed_pairs)} kayıt bulundu, atlanacak.")

    total_written = 0
    t_start = time.time()

    for v_idx, (video_path, ground_truth) in enumerate(videos, 1):
        v_name = os.path.basename(video_path)

        for cfg in top_configs:
            pair_key = (str(cfg["id"]), v_name)
            if pair_key in processed_pairs:
                continue

            try:
                res = run_config_on_video(
                    video_path, cfg, model, BENCHMARK_OUT_DIR,
                    max_frames, False, save_video=False,
                    ground_truth=ground_truth, verbose=False
                )
                confusion = res.get("confusion", "")
                if confusion == "TP": leaderboard[str(cfg["id"])]["TP"] += 1
                elif confusion == "FN": leaderboard[str(cfg["id"])]["FN"] += 1
                elif confusion == "TN": leaderboard[str(cfg["id"])]["TN"] += 1
                elif confusion == "FP": leaderboard[str(cfg["id"])]["FP"] += 1
                leaderboard[str(cfg["id"])]["processed_videos"].append(v_name)
                processed_pairs.add(pair_key)
                total_written += 1
            except Exception as e:
                print(f"  ❌ C{cfg['id']:02d} | {v_name}: {e}")
            finally:
                cv2.destroyAllWindows()

        elapsed = time.time() - t_start
        rate = v_idx / max(elapsed, 1)
        remaining = (len(videos) - v_idx) / max(rate, 0.001)
        print(f"\r  📊 [{v_idx}/{len(videos)}] vid | {total_written} CSV satır | "
              f"{elapsed/60:.0f}dk geçti | ~{remaining/3600:.1f}sa kaldı     ", end="", flush=True)

        # Her 50 videoda leaderboard'u kaydet
        if v_idx % 50 == 0:
            save_leaderboard(leaderboard)

    save_leaderboard(leaderboard)
    cv2.destroyAllWindows()
    print(f"\n\n✅ Hızlı vektör modu tamamlandı! {total_written} yeni satır eklendi.")
    print(f"   CSV: {VECTOR_TRAIN_CSV}")


    print("\n" + "═" * 80)
    print("  [MODE 5] VERİTABANI SIFIRLA")
    print("═" * 80)
    confirm = input("Tüm Leaderboard ve Vektör Logları silinecek! 'SIFIRLA' yazın: ").strip()
    if confirm == "SIFIRLA":
        if os.path.exists(LEADERBOARD_JSON): os.remove(LEADERBOARD_JSON)
        if os.path.exists(VECTOR_TRAIN_CSV): os.remove(VECTOR_TRAIN_CSV)
        print("✅ Veritabanı sıfırlandı. Artık baştan test edebilirsiniz.")
    else:
        print("❌ İptal edildi.")

def main():
    while True:
        print("\n" + "═" * 80)
        print("  ADAS BENCHMARK SUITE v2.0  —  DOĞRULUK & VECTOR ML TEST MOTORU")
        print("═" * 80)
        print(" [1] Toplu Benchmark Çalıştır (Pozitif/Negatif Kesintisiz Besleme)")
        print(" [2] Leaderboard ve Doğruluk (Accuracy) Tablosu")
        print(" [3] 🏆 Top Performers Benchmark (5 Kötü Config Elenmiş — Daha Hızlı)")
        print(" [4] ⚡ Hızlı Vektör Modu (150 frame, no-video, sadece CSV log)")
        print(" [5] 🗑️  Veritabanını Sıfırla (Kaldığın yeri unut ve baştan başla)")
        print(" [0] Çıkış")
        print("═" * 80)

        choice = input("\nİşlem seçin: ").strip()

        if choice == "1": mode_benchmark_batch()
        elif choice == "2": mode_comparison_leaderboard()
        elif choice == "3": mode_benchmark_top_performers()
        elif choice == "4": mode_fast_vector()
        elif choice == "5": mode_reset_database()
        elif choice == "0": sys.exit(0)

if __name__ == "__main__":
    main()