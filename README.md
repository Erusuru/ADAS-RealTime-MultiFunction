# 🚗 Real-Time, Multi-Function ADAS Application

[![Platform](https://img.shields.io/badge/hardware-Raspberry%20Pi%205%20%7C%20Hailo--10H-red.svg)](https://www.raspberrypi.com/products/raspberry-pi-5/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/Erusuru/ADAS-RealTime-MultiFunction?style=social)](https://github.com/Erusuru/ADAS-RealTime-MultiFunction/stargazers)
[![Issues](https://img.shields.io/github/issues/Erusuru/ADAS-RealTime-MultiFunction)](https://github.com/Erusuru/ADAS-RealTime-MultiFunction/issues)
[![Last Commit](https://img.shields.io/github/last-commit/Erusuru/ADAS-RealTime-MultiFunction)](https://github.com/Erusuru/ADAS-RealTime-MultiFunction/commits/main)

**Author:** Ramazan Ertuğrul Aydoğan  
**Affiliation:** *Department of Electrical and Electronics Engineering, Gaziantep University (GAÜN), Gaziantep, Türkiye*

**Co-author:** Fatima Sapundzhi — *South-West University "Neofit Rilski", Blagoevgrad, Bulgaria*

⭐ **If this project is useful or interesting to you, consider starring the repo — it helps others find it.**

---

## 📚 Table of Contents

- [Why This Project](#-why-this-project)
- [Overview](#-overview)
- [Crash & Near-Miss Scenario Gallery](#-comprehensive-crash--near-miss-scenario-gallery)
- [Video Demonstrations](#-video-demonstrations)
- [System Architecture](#️-system-architecture)
- [Performance Benchmarks (113 Scenarios)](#-performance-benchmarks-113-manually-reviewed-scenarios)
- [Automated Benchmark (2,844 Nexar Sequences)](#-automated-benchmark-2844-nexar-derived-sequences)
- [Custom-Trained YOLO12s Models](#-custom-trained-yolo12s-models)
- [Android Companion App](#-android-companion-app)
- [Bill of Materials](#-bill-of-materials-bom)
- [Future Work](#-future-work)
- [Related Publication](#-related-publication)
- [License](#-license)

---

## 💡 Why This Project

Most older vehicles lack built-in driver-assistance technology, since factory-installed ADAS remains expensive and closed. This project is an **open-source, low-cost ADAS suite** that keeps hardware cost down while remaining installable on older vehicles — acting as a co-pilot for the two most safety-critical maneuvers: **forward driving** and **reverse parking**.

---

## 📌 Overview

This repository documents the architecture, visual evaluation, and benchmarking of a **state-driven, six-function Advanced Driver-Assistance System (ADAS)** designed for low-cost embedded edge platforms. The intended deployment target is a **Raspberry Pi 5** paired with a **Hailo-10H AI Accelerator** (rated up to **40 TOPS**, INT4). That target board was unavailable during this study period due to the global memory-chip shortage, so the system was validated on substitute hardware instead:

- **Raspberry Pi 4B (4 GB RAM):** LDW, DMS, ultrasonic parking assist, automatic headlight control, and the YOLO11n baseline detector.
- **Laptop tiers (Intel i5-8265U/MX110 and AMD Ryzen 7 AI 350/RTX 5060, 32 GB RAM):** the heavier perception stack — a custom-trained YOLO12s detector plus the vector-based Forward Collision Warning (FCW) module.

The architecture integrates six core active-safety and convenience functions, coordinated by a central state controller that switches between `FORWARD_MODE` and `REVERSE_MODE` based on the vehicle's reverse-gear signal:

1. **Lane Departure Warning (LDW):** Classical CV pipeline — grayscale conversion, Gaussian filtering, Canny edge detection, and Probabilistic Hough Transform (OpenCV) within a trapezoidal ROI.
2. **YOLO Object Detection + Deterministic Vector-Based Forward Collision Warning (FCW):** YOLO11n (Pi 4B tier) or custom-trained YOLO12s (laptop tiers) feeds a ByteTrack-refined, six-state Kalman filter. The physics layer tracks centroid history over 10 frames, computes image-plane velocity vectors, projects a ~1s trajectory, and checks intersection against the ego-lane polygon — flagging both longitudinal rear-end threats and lateral cut-in / T-bone threats that a purely longitudinal check would miss. Monocular distance is estimated via a calibrated pinhole model (`Distance = Focal Length × Real Width / Pixel Width`).
3. **Automatic Reverse Assist & Ultrasonic Parking:** Reverse-gear-triggered rear-camera feed with distance-modulated buzzer feedback via four JSN-SR04T ultrasonic sensors.
4. **Intelligent Automatic Headlight Control:** MCP3008 ADC reading ambient-light levels with dual-threshold hysteresis, controlling an isolated 12V automotive relay.
5. **Driver Monitoring System (DMS):** MediaPipe Face Mesh landmark tracking computing Eye Aspect Ratio (EAR) from six landmarks per eye. EAR below **0.22 for more than 1.5s** triggers the audible drowsiness warning; a longer 5s persistence window was used specifically for the physical actuator-cutoff bench demo.
6. **Simulation-in-the-Loop (SITL) Bridge:** High-level threat events are mapped to steering/braking commands in **BeamNG.tech** for closed-loop AEB and driver-monitoring validation. SAE Level 2-style closed-loop steering/braking is exercised **only inside BeamNG.tech**; on the real hardware the system performs SAE Level 0 driver warning only — there is no mechanical actuation on a physical vehicle, since integrating ADAS actuation onto a real car was outside the scope of this project.

DMS, ambient-light sensing, and diagnostics remain active in both operational modes.

---

## 📸 Comprehensive Crash & Near-Miss Scenario Gallery

The system was evaluated across **113 diverse real-world crash and near-miss scenarios** (day, night, rain, dust). Below is the verified visual gallery detailing the system's detection and warning performance across all accident typologies:

### 1. 🛑 Longitudinal Forward Collisions & Lead Vehicle Deceleration (94.0% Timely-Warning Rate)

| Lead SUV Close Proximity (`2.9m`) | High-Speed Sun Glare (`107 km/h`) | Residential Two-Lane Approach |
|:---:|:---:|:---:|
| ![Lead SUV Proximity](assets/images/longitudinal_01_lead_suv_proximity_2.9m.jpg) | ![Direct Sun Glare](assets/images/longitudinal_02_direct_sun_glare_107kmh.jpg) | ![Residential Road Approach](assets/images/longitudinal_03_residential_road_approach.jpg) |
| **`car 2.9m`** in RED box on lead Mitsubishi Outlander SUV triggering **`RED - EMERGENCY BRAKE`**. | High-speed highway following at **107 km/h** driving directly into low-angle blinding sun glare (**`car 10.7m`**). | Two-lane suburban roadway approach to lead vehicle (**`car 12.0m`**) with ego corridor overlay. |

| Overcast Morning Deceleration | Oncoming Centerline Drift | Rural Driveway Chrysler Approach |
|:---:|:---:|:---:|
| ![Overcast Slowdown](assets/images/longitudinal_05_overcast_morning_slowdown.jpg) | ![Centerline Drift](assets/images/longitudinal_06_oncoming_centerline_drift.jpg) | ![Rural Roadway Approach](assets/images/longitudinal_07_rural_roadway_approach.jpg) |
| Overcast morning highway lead vehicle rapid slowdown (**`car 7.0m`** at 27 mph). | Oncoming vehicle crossing center double yellow lines into host drivable corridor (**`car 8.4m`**). | Rural road approach to oncoming/turning Chrysler sedan (**`car 5.7m`**) with trajectory vector line. |

| Multi-Lane Desert Highway Approach |
|:---:|
| ![Desert Highway](assets/images/longitudinal_08_multi_lane_desert_highway.jpg) |
| Wide desert highway approach to lead vehicle with active green ego corridor. |

---

### 2. ⚡ Lateral Cross-Traffic & Intersection T-Bone Hazards (48.3% Timely-Warning Rate)

| Signalized Intersection Cross-Traffic | Intersection Impact / Damaged Hood | Urban Avenue Red Beetle Crossing |
|:---:|:---:|:---:|
| ![Signalized Intersection](assets/images/tbone_01_signalized_intersection_crossing.jpg) | ![Damaged Truck Impact](assets/images/tbone_02_intersection_damaged_truck_impact.jpg) | ![Red Beetle Crossing](assets/images/tbone_03_urban_avenue_red_beetle_crossing.jpg) |
| Black sedan traversing signalized intersection perpendicularly (**`car 6.2m`**) under active traffic lights. | Blue Ford pickup truck with crumpled hood at intersection crossing (**`car 1.9m`**). | Red VW Beetle crossing urban commercial avenue perpendicularly from left (**`car 3.9m`**). |

| Crossroad Red Sedan Incursion | Range Rover Perpendicular Crossing | Commercial Driveway Pull-Out |
|:---:|:---:|:---:|
| ![Crossroad Incursion](assets/images/tbone_04_crossroad_red_sedan_incursion.jpg) | ![Grey SUV Crossing](assets/images/tbone_05_perpendicular_grey_suv_crossing.jpg) | ![Commercial Driveway Pullout](assets/images/tbone_06_commercial_driveway_pullout.jpg) |
| Red sedan traversing perpendicular crossing path under green traffic light (**`car 10.4m`**). | Grey Range Rover SUV crossing host vehicle's drivable path perpendicularly from left (**`car 3.8m`**). | Dark hatchback pulling out perpendicularly from parking lot/commercial entrance on right (**`car 4.5m`**). |

---

### 3. 🔀 Lateral Merging, Cut-Ins & Vehicle Incursions (27.3% Timely-Warning Rate)

| Aggressive Perpendicular Merge | Highway Ramp Merging Incursion | Blind-Spot Close Incursion | Rural Highway Crossroad Pull-Out |
|:---:|:---:|:---:|:---:|
| ![White Car Cut-In](assets/images/cutin_01_perpendicular_merge_white_car.jpg) | ![Highway Ramp Sedan](assets/images/cutin_02_highway_ramp_merging_sedan.jpg) | ![Blind Spot Incursion](assets/images/cutin_03_blind_spot_close_incursion.jpg) | ![Rural Red SUV Pullout](assets/images/cutin_04_rural_roadside_red_suv_entry.jpg) |
| **`car 1.8m [T-BONE?]`** cutting into host lane; flagged 500ms before crossing lane markings. | Dark sedan merging into curved highway ramp from right shoulder (**`car 4.7m`**). | Close-proximity vehicle cutting closely in front of bumper (**`car 1.6m [T-BONE?]`**). | Red SUV pulling out perpendicularly from right roadside at 49 mph (**`car 2.6m`**). |

---

### 4. 🌙 Adverse Lighting, Low-Light & Impact Scenarios (8.3% Timely-Warning Rate)

| Nighttime Urban Streetlight Driving | Rear-End Impact Bumper Damage |
|:---:|:---:|
| ![Night Urban Driving](assets/images/adverse_01_night_urban_intersection_streetlight.jpg) | ![Bumper Impact Damage](assets/images/adverse_03_rear_end_bumper_impact_underpass.jpg) |
| True nighttime urban driving at 52 km/h under streetlights; turning white SUV (**`car 4.4m`**; HUD: **`NIGHT \| conf 0.25`**). | Immediate proximity to damaged Chevy Silverado tailgate under highway overpass (**`car 1.6m`**). |

---

### 5. 🏙️ Dense Multi-Target Urban Environments & HUD Telemetry

| Dense Palm-Tree Avenue Crosswalk | Residential Driveway Parked Vehicles |
|:---:|:---:|
| ![Dense Avenue Crosswalk](assets/images/multi_target_01_dense_avenue_pedestrian_crosswalk.jpg) | ![Residential Parked Cars](assets/images/multi_target_02_residential_parked_cars.jpg) |
| High-density urban avenue approaching pedestrian crosswalk with **7 surrounding vehicles tracked simultaneously** (**`car 11.6m`** lead alert). | Residential neighborhood drive with parked vehicles (**`car 14.5m`** lead alert with **`car 7.4m`** parked SUV in orange). |

---

### 6. 🎮 Simulation-in-the-Loop (BeamNG.tech AEB Validation)

> ⚠️ **SAE Level applies to simulation only.** The SAE Level 2-style closed-loop steering/braking below is exercised **exclusively inside BeamNG.tech**. On the actual Raspberry Pi hardware, the system is **SAE Level 0** — it only warns the driver; there is no mechanical actuation on a real vehicle, as adapting ADAS actuation onto a physical car was outside the scope of this project (no team/resources for that integration).

<img src="assets/images/GMBH-Logo.png" alt="BeamNG GmbH" width="130"/>

Threat events are bridged from the perception stack into **BeamNG.tech**, which maps them to steering or braking commands for closed-loop validation. In a deliberately forced over-speeding intersection test — a van approaching a 50 km/h zone at 90 km/h behind a braking lead vehicle — the system detected the stationary hazard and applied full AEB, reducing impact speed from **90 km/h to 35 km/h (≈84.9% of kinetic energy dissipated)**. A corresponding drowsiness event from the DMS is also mapped to braking in BeamNG.tech simulations.

| Level 2 AEB Emergency Stop (BeamNG.tech simulation only) | Drivable Corridor & Closed-Loop Centering (BeamNG.tech simulation only) |
|:---:|:---:|
| ![BeamNG AEB Stop](assets/images/sitl_01_beamng_level2_aeb_emergency_stop.jpg) | ![BeamNG Autopilot Corridor](assets/images/sitl_02_beamng_drivable_corridor_autopilot.jpg) |
| Autopilot Mode 2 (Braking Only) executing Automated Emergency Braking (AEB) upon collision vector detection — simulated in BeamNG.tech, not on real hardware. | Closed-loop lane centering and road curvature drivable corridor segmentation overlay — simulated in BeamNG.tech, not on real hardware. |

---

## 🎬 Video Demonstrations

Curated demonstration clips are located in [`assets/videos/`](assets/videos/):
- 🎥 [`demo_fcw_longitudinal_rear_end.mp4`](assets/videos/demo_fcw_longitudinal_rear_end.mp4) — High-speed highway following and imminent braking.
- 🎥 [`demo_fcw_lateral_tbone_threat.mp4`](assets/videos/demo_fcw_lateral_tbone_threat.mp4) — Urban intersection cross-traffic alert.
- 🎥 [`demo_fcw_merging_cutin_threat.mp4`](assets/videos/demo_fcw_merging_cutin_threat.mp4) — Aggressive lateral cut-in threat detection.
- 🎥 [`demo_fcw_intersection_imminent_brake.mp4`](assets/videos/demo_fcw_intersection_imminent_brake.mp4) — Head-on intersection collision avoidance.
- 🎥 [`demo_sitl_beamng_level2_autopilot.mp4`](assets/videos/demo_sitl_beamng_level2_autopilot.mp4) — BeamNG.tech SITL Level 2 closed-loop intervention (target-occlusion and crash-energy mitigation test, 90→35 km/h).

---

## 🏛️ System Architecture

The software architecture is structured as a concurrent multi-threaded finite state machine (FSM), coordinated by a central state controller that arbitrates camera/display resources between `FORWARD_MODE` and `REVERSE_MODE`:

```mermaid
flowchart TD
    subgraph Hardware Layer
        CAM_FWD[Forward USB 1080p Camera]
        CAM_REAR[Rear Bumper USB 720p Camera]
        CAM_DMS[Driver-Facing Camera]
        SONAR[JSN-SR04T Waterproof Ultrasonic Sensors]
        LDR[LDR Photoresistor + MCP3008 ADC]
        GPIO_REV[12V Reverse Light Signal -> Optocoupler]
        PI5[Raspberry Pi 5 + Hailo-10H NPU]
    end

    subgraph State Machine Controller
        FSM{Main Control Thread\nPoll Reverse GPIO}
    end

    subgraph FORWARD_MODE [FORWARD_MODE Threads]
        LDW[Lane Departure Warning\nCanny + Hough]
        YOLO[YOLO11n / YOLO12s Object Detector\n+ ByteTrack]
        FCW[Vector Physics Engine\nKalman Filter + TTC + Cut-In + Ego Corridor]
        DMS[Driver Monitoring\nMediaPipe Face Mesh + EAR]
        BSM[Blind Spot Ultrasonic Monitor\n<= 3.0m Side LEDs]
        HEADLIGHT[Auto Headlight Controller\nHysteresis 30%/50%]
    end

    subgraph REVERSE_MODE [REVERSE_MODE Threads]
        FFPLAY[Low-Latency Rear Camera Stream]
        PARK[Parking Distance Sonar + Buzzer Modulation]
    end

    GPIO_REV --> FSM
    FSM -- FORWARD --> FORWARD_MODE
    FSM -- REVERSE --> REVERSE_MODE

    CAM_FWD --> LDW & YOLO
    YOLO --> FCW
    LDW --> FCW
    CAM_DMS --> DMS
    SONAR --> BSM & PARK
    LDR --> HEADLIGHT
    CAM_REAR --> FFPLAY
```

---

## 📊 Performance Benchmarks (113 Manually Reviewed Scenarios)

The system was evaluated across **113 diverse real-world crash and near-miss scenarios** spanning highway, urban, low-light, and adverse weather conditions. A warning was considered timely if issued at least **1.5s** before the annotated impact or critical conflict point.

### Table II: Scenario-Specific Detection Accuracy

| Scenario Type | Total ($N$) | Timely Warnings | Late / Missed | Timely-Warning Rate (95% CI) |
|:---|:---:|:---:|:---:|:---:|
| **Longitudinal (Rear-End)** | 50 | 47 | 3 | **94.0%** (83.8–97.9%) |
| **Lateral (T-Bone / Intersection)** | 29 | 14 | 15 | **48.3%** (31.4–65.6%) |
| **Lane Cut-Ins (Partial Object)** | 22 | 6 | 16 | **27.3%** (13.2–48.2%) |
| **Nighttime / Low-Light** | 12 | 1 | 11 | **8.3%** (1.5–35.4%) |
| **TOTAL** | **113** | **68** | **45** | **60.2%** (51.0–68.7%) |

- **Nuisance Alerts:** **5 of 113 cases (4.4%)** — occurring exclusively when adjacent vehicles overtook the host car at close lateral distance. This is not a time-normalized false-alarm rate; specificity was not calculated on the manually reviewed set.

---

## 🤖 Automated Benchmark (2,844 Nexar-Derived Sequences)

In addition to the manual 113-scenario review, seven vision-pipeline configurations were compared across 2,844 processed sequences derived from the [Nexar dashcam collision dataset](https://arxiv.org/abs/2503.03848).

**<font color="red">Because the automated scoring script flags any alert without an actual crash as a false positive — and the Nexar set is mostly near-miss footage where drivers avoided a crash — the automated false-alarm rate is not directly comparable to the low nuisance-alert rate measured in manual review above.</font>**

The **recall/timely-warning numbers below are trustworthy** — a correct warning issued in time is a correct warning whether or not a crash actually followed. It's specifically the *false-alarm* percentage in the automated benchmark that is inflated and not reflective of real-world nuisance-alert behavior; the manually reviewed 4.4% nuisance-alert rate above is the accurate figure for that.

| Configuration | Recall / Sensitivity | False Alarm Rate | F1-Score | Nighttime Acc. |
|:---|:---:|:---:|:---:|:---:|
| **Config 01 — CPU baseline** (raw pinhole distance) | 89.80% | 77.08% | 67.69% | 60.32% |
| **Config 11 — balanced optimum** (conf=0.35) | 90.72% | 76.04% | **68.22%** (highest F1) | 63.49% |
| **Config 17 — high recall** (TwinLiteNet, 640×384) | **92.50%** (highest recall) | 80.83% | 67.68% | **63.64%** (highest night acc.) |

---

## 🧠 Custom-Trained YOLO12s Models

Two 9.26-million-parameter **YOLO12s** models were trained at 1024×1024 resolution on an RTX 5060 laptop GPU, forming the heavier detection tier used for the vector-based FCW module:

| Model | Classes | Training | Validation mAP@50 | Validation mAP@50-95 |
|:---|:---:|:---|:---:|:---:|
| **Turkish Traffic Sign Detector** | 24 | 85 epochs (50 initial + 35 fine-tuning) | **93.47%** | 73.11% |
| **Self-Driving / Road-Object Detector** (7-class) | 7 | Reduced Roboflow-hosted base set + custom dashcam additions | **94.75%** | — |

These are training-log validation metrics. The lighter **YOLO11n** baseline detector (no custom training) is used on the Raspberry Pi 4B tier where the full YOLO12s stack is too heavy.

---

## 📱 Android Companion App

A beta-stage Android app (Kotlin / Jetpack Compose) brings the ADAS perception stack to smartphones, independent of the embedded Raspberry Pi platform:

- **CameraX + TensorFlow Lite** inference with a GPU → NNAPI → CPU delegate cascade for on-device fallback.
- **11 detected classes** with class-aware Non-Maximum Suppression (NMS).
- A **SAFE / CAUTION / WARNING / CRITICAL** threat classifier with a live time-to-collision (TTC) estimate.
- A separate automated benchmarking pipeline uses a large multimodal model as an objective judge — scoring bounding-box validity, threat severity, and warning timing to produce reward scores for future training data.

<img src="assets/images/android_app_screenshot.jpg" alt="Android ADAS companion app" width="280"/>

---

## 💰 Bill of Materials (BOM)

### Table I: Prototype Hardware Implementation

> ⚠️ The Raspberry Pi 5 + Hailo-10H target configuration below was not benchmarked in the current study (hardware unavailable during this cycle). See [Overview](#-overview) for the substitute platforms actually used for validation.

| Component | Specification | Est. Cost (EUR) |
|:---|:---|:---:|
| **Compute Module** | Raspberry Pi 5 (16 GB RAM) + Active Cooler | €145 |
| **AI Acceleration** | Hailo-10H M.2 Module (~40 TOPS, INT4) + M.2 HAT+ Adapter | €140 |
| **Visual Sensors** | 2× USB 1080p Wide-Angle Cameras + 1× USB 720p Rear Camera | €100 |
| **Display** | 7-inch Capacitive Touchscreen (DSI/HDMI) | €65 |
| **Proximity Sensors** | 4× JSN-SR04T Waterproof Ultrasonic Sensors | €45 |
| **Power Management** | 12V-to-5V 5A Buck Converter + Optoisolated Relay Module | €25 |
| **Peripherals** | MCP3008 ADC, Wiring Harness, Enclosure, PCB | €30 |
| **TOTAL** | | **€545** |

---

## 🔭 Future Work

- **Hailo-10H hardware validation** — full six-subsystem integration and benchmarking on the target Raspberry Pi 5 + Hailo-10H platform once hardware is available.
- **Luminance-adaptive segmentation** using **TwinLiteNetPlus** to improve low-light/nighttime performance.
- **Expanded smartphone-based inference** — building out the Android companion app further.
- **CAN-bus vehicle telemetry integration.**
- **YOLO road-hazard model** for detecting animals, potholes, speed bumps, and other road-surface anomalies.

---

## 📄 Related Publication

This repository accompanies the following peer-reviewed proceedings paper:

> **Design and Simulation Evaluation of an Embedded Multi-Sensor ADAS Architecture with AI-Assisted Collision Warning**  
> Ramazan Ertuğrul Aydoğan, Fatima Sapundzhi  
> Presented at the 13th International Electronic Conference on Sensors and Applications (ECSA-13), 18–20 November 2026  
> *Engineering Proceedings* (MDPI)

### Citation

DOI and article link are not yet assigned (pending MDPI production) — update the fields below once available.

```bibtex
@inproceedings{aydogan2026adas,
  author    = {Aydoğan, Ramazan Ertuğrul and Sapundzhi, Fatima},
  title     = {Design and Simulation Evaluation of an Embedded Multi-Sensor ADAS Architecture with AI-Assisted Collision Warning},
  booktitle = {Proceedings of the 13th International Electronic Conference on Sensors and Applications (ECSA-13)},
  series    = {Engineering Proceedings},
  publisher = {MDPI},
  year      = {2026},
  month     = {11},
  note      = {DOI to be assigned},
  url       = {}
}
```

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

![Visitors](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FErusuru%2FADAS-RealTime-MultiFunction&count_bg=%23E34C26&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false)

</div>
