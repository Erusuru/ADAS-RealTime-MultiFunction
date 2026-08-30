**Real-Time, Multi-Function ADAS Application**

Ramazan Ertuğrul Aydoğan

*South-West University Neofit Rilski, Blagoevgrad, Bulgaria*

*Received: January 2, 2020 • Revised: February 4, 2020 • Accepted: April 22, 2020 • Published: June 9, 2020*

***Abstract***—This research covers the design, implementation, and evaluation of a comprehensive, real-time Advanced Driver-Assistance System (ADAS) engineered for low-cost embedded platforms. Six safety and convenience features operate concurrently on a single Raspberry Pi 5 augmented by a Hailo-8 AI accelerator: a Lane Departure Warning (LDW) system, real-time YOLO-based object detection, an automated rear-view camera, an ultrasonic parking sensor, an intelligent automatic headlight system, and a driver-facing facial landmark tracker computing the Eye Aspect Ratio (EAR) for drowsiness detection. A state-driven, multi-threaded software model allocates computational resources based on the vehicle’s operational state. A deterministic Forward Collision Warning (FCW) algorithm combines YOLO tracking with vector-based physics to predict collisions—including cut-ins and T-bone threats—before they occur. System validation across 113 crash and near-miss scenarios achieved a 94% detection rate for longitudinal rear-end events and 60.1% overall. Simulation-in-the-Loop (SITL) testing with BeamNG.drive validated Level 2 active intervention capabilities including Automatic Emergency Braking (AEB). A parallel development effort has produced a native Android ADAS application in Kotlin, currently in development at lower detection accuracy than the embedded Python baseline, targeting commodity smartphones via TFLite inference. An automated AI training data acquisition pipeline with human-in-the-loop reward scoring has been established to support future model improvement.

***Keywords:*** *Advanced Driver-Assistance Systems (ADAS), Lane Departure Warning, Forward Collision Warning, Eye Aspect Ratio, YOLO, Computer Vision, TFLite, Android, Sensor Fusion, Real-Time Systems, Raspberry Pi, Hailo-8, JSN-SR04T.*

**I. Introduction**

Every year, preventable traffic accidents result in a significant number of injuries, fatalities, and economic damage. Human error remains the predominant causal factor in the majority of collisions worldwide. In response, the automotive industry has developed Advanced Driver-Assistance Systems (ADAS), a class of technologies designed to augment driver perception and mitigate the risks associated with distraction, fatigue, and momentary lapses in attention. While lane-keeping assistance, forward-collision warning, and parking assistance are increasingly standard in new vehicles, a substantial proportion of older vehicles remain unequipped with these potentially life-saving technologies.

The high cost and proprietary, closed nature of factory-installed ADAS present a significant barrier to third-party adoption. This research directly addresses this accessibility gap by engineering an open-source, fully integrated ADAS system at minimal cost, designed to operate across a wide variety of older vehicles.

The primary objective is to develop an all-in-one co-pilot system for two of the most frequent and critical driving scenarios: high-speed forward driving and precise reverse parking. Six integrated subsystems were developed: (1) Lane Departure Warning (LDW); (2) real-time Object Detection; (3) Automatic Reverse Camera; (4) Ultrasonic Parking Assist; (5) Intelligent Automatic Headlight Control; and (6) a Driver Monitoring System (DMS) for fatigue detection. The methodological foundation is a pragmatic hybrid approach combining classical computer vision for geometrically well-defined tasks with deep learning for complex object recognition, orchestrated on a Raspberry Pi 5.

In parallel, a native Android ADAS application and an AI training data acquisition pipeline have been developed to extend the system’s reach and support future model improvement through human-in-the-loop reward scoring.

**II. System Architecture and Methodology**

The architecture constitutes a multi-layer construct of physical hardware, open-source software, and Python libraries. The core algorithm interprets real-time data from cameras and sensors and translates it into actionable driver warning feedback.

**A. Hardware Platform**

The Raspberry Pi 5 serves as the central compute module, featuring a 64-bit quad-core ARM Cortex-A76 processor and VideoCore VII GPU. The 26 TOPS Hailo-8 AI Accelerator connects via PCIe as an add-on HAT, offloading AI model inference from the CPU while preserving full GPIO functionality through a stacking pass-through header. The combination delivers real-time deep learning inference at low power on a compact, affordable platform.

Two standard USB wide-angle cameras provide visual input. The forward-facing camera requires a minimum 90-degree field of view to capture roadside signage on multi-lane roads; overly narrow cameras (45–60 degrees FOV) will fail to detect signs on adjacent lanes. The rear camera, mounted at the vehicle’s bumper, should use a moderate FOV; cameras exceeding 120 degrees introduce perspective distortion that may cause drivers to misjudge proximity to obstacles. JSN-SR04T waterproof ultrasonic sensors—featuring hermetically sealed probes with 2.5 m cables—are used for proximity sensing during reverse maneuvers and blind-spot monitoring, allowing external bumper mounting while protecting electronics inside the chassis.

Driver feedback is delivered through directional LED warning lights indicating lane drift direction and a central audible buzzer generating distinct alert patterns for each warning type. The system’s operational state is controlled by a GPIO input pin wired directly to the vehicle’s reverse light circuit. Because the reverse light operates at 12 V, the signal must be reduced to the Raspberry Pi’s 3.3 V GPIO logic level using a resistor voltage divider or a 3.3 V-rated optocoupler.

An MCP3008 Analog-to-Digital Converter (ADC) interfaces via SPI bus to read the LDR photoresistor for headlight control. A 12 V automotive relay (30 A rated) bridges the GPIO and the vehicle’s electrical system, controlled via an optocoupler for back-EMF isolation. A 7-inch capacitive LCD touchscreen (HDMI/DSI) serves as the primary display for the rear camera feed and visual alerts.

**B. Software and Core Technologies**

The system is implemented in Python, leveraging the threading library for true concurrency. All ADAS subsystems run in independent threads, ensuring no single task can block another’s execution. OpenCV provides the complete lane detection pipeline from frame capture and preprocessing through Hough-based line detection. NumPy enables efficient vectorized operations on pixel data. Object detection employs the Ultralytics YOLO framework, which treats detection as a single regression problem: in one network forward pass it predicts all bounding boxes, class probabilities, and confidence scores for an entire frame, achieving real-time throughput on resource-constrained hardware. Hardware peripheral control uses the RPi.GPIO library.

**C. Core Algorithms and Processing Pipelines**

***Lane Departure Warning Branch***

Each camera frame is preprocessed through a multi-stage pipeline: gamma correction reduces overexposure in high-contrast conditions; CLAHE contrast enhancement improves marking visibility; HLS-space color filtering isolates white and yellow lane markings; grayscale conversion and Gaussian blur suppress noise. The Canny Edge Detector applies dual-threshold hysteresis to distinguish genuine edges from noise. A trapezoidal Region of Interest (ROI) mask constrains processing to the vehicle’s current lane, excluding adjacent lanes and guardrails. The Probabilistic Hough Transform maps edge points to parameter space where collinear points converge, enabling efficient line segment detection. Segments are grouped by slope, averaged, and smoothed using an exponential moving average (α = 0.8) to produce stable left and right lane boundary lines. Lateral deviation exceeding 5% of screen width triggers a directional warning.

***Object Detection Branch***

Concurrently, the original frame is forwarded to a pre-trained YOLOv8n model. Each detection yields a class identifier, confidence score, and bounding box coordinates. Detections below a confidence threshold of 0.4 are discarded. Labeled bounding boxes are rendered in debug mode only; the driver receives alerts solely when the physics engine flags specific threat conditions.

***Traffic Sign Recognition Branch (Theoretical Implementation)***

The TSR component employs a two-stage pipeline. The primary YOLOv11s model performs localization, detecting a “traffic\_sign” super-class bounding box. The cropped ROI is then passed to a secondary lightweight CNN for specific sign classification. Key challenges include scale invariance (signs may occupy fewer than 32×32 pixels at high speed) and domain adaptation between European, Turkish, and local regulatory standards, necessitating a modular retraining pipeline.

***Reverse Assist Pipeline***

Upon GPIO reverse-signal detection, the system launches an ffplay subprocess to display the rear camera feed at minimal latency. Lane detection and sign recognition are suspended, releasing computational resources. The ultrasonic sensor thread continuously measures obstacle distance and modulates the buzzer’s beep frequency proportionally to proximity, providing an intuitive auditory parking aid.

***Automatic Headlight Logic***

The system reads ambient light continuously from the ADC-connected LDR and applies a hysteresis algorithm to prevent flickering. The Turn-ON threshold activates headlights when light falls below 30% for more than 2 consecutive seconds. The Turn-OFF threshold deactivates lights only when light exceeds 50% for more than 2 seconds. The relay is wired in parallel with the vehicle’s existing headlight switch, with optocoupler isolation between the GPIO and relay coil to protect against back-EMF voltage spikes.

**D. System Operational Logic and State Management**

The system operates as a finite state machine with two primary states: FORWARD\_MODE and REVERSE\_MODE. The Main Control Thread continuously polls the GPIO reverse-signal pin. State transitions are handled as follows:

* **Transition to REVERSE\_MODE:** Forward Processing Thread is terminated; rear-camera subprocess and Parking Sensor Thread are launched.
* **Transition to FORWARD\_MODE:** Rear-camera subprocess is terminated; Forward Processing Thread, Blind Spot Monitor Thread, and Auto Headlight Thread are re-initialized.

PROCEDURE MAIN\_SYSTEM\_LOOP

Initialize System Resources; Load AI Models onto CPU/NPU

Initialize Threading Locks and Shared Memory

WHILE System is Active DO:

Read State of Reverse\_Gear\_GPIO

IF State = REVERSE\_MODE THEN:

Stop FORWARD\_ADAS\_THREAD

Spawn/Wake REVERSE\_ASSIST\_THREAD

ELSE IF State = FORWARD\_MODE THEN:

Stop REVERSE\_ASSIST\_THREAD

Spawn/Wake FORWARD\_ADAS\_THREAD

Spawn/Wake BLIND\_SPOT\_MONITOR\_THREAD

Spawn/Wake AUTO\_HEADLIGHT\_THREAD

Sleep(10 ms) // Prevent CPU hogging

END PROCEDURE

***Blind Spot Detection Pipeline***

A dedicated background thread sequentially triggers the JSN-SR04T sensors on the vehicle’s rear flanks, computing time-of-flight using the speed of sound at sea level (343.2 m/s). Detection of an object within the configurable danger zone (default: ≤3 m) during forward motion activates the corresponding side-mirror warning LED, independent of all other threads.

***Vector-Based Forward Collision Warning Branch***

A physics-based tracking layer operates on top of the YOLO architecture in four stages:

* **Historical Trajectory Tracking:** A Deque structure stores centroid coordinates (cx, cy) and bounding box width for each tracked vehicle across the last 10 frames.
* **Velocity Vector Calculation:** Position changes (Δx, Δy) over elapsed time (Δt) yield velocity vectors (vx, vy), distinguishing static vehicles from those approaching or accelerating toward the host vehicle.
* **Future Path Prediction:** Velocity vectors project each object’s position approximately one second ahead. A polygon intersection test determines whether the predicted position falls within the trapezoidal Ego Lane safety zone.
* **Lateral Threat Detection (T-Bone Logic):** Objects in the middle vertical band (30–80% of screen height) exhibiting high lateral velocity (vx) directed toward the screen center are flagged as cut-in or T-bone threats, triggering an immediate alert regardless of current distance.

PROCEDURE VECTOR\_BASED\_FCW\_LOGIC

INPUT: Current Frame Detections (Bounding Boxes, Object IDs)

CONSTANTS: HISTORY\_LENGTH=10; LATERAL\_SPEED\_THETA=2.0

FOR EACH Detected Object DO:

Store (cx, cy, width, time) in Track\_History[Object\_ID]

IF Length(Track\_History) >= 5 THEN:

Compute vx = Delta(cx)/Delta(t) ; vy = Delta(cy)/Delta(t)

approach\_speed\_ms = (old\_dist - new\_dist) / Delta(t)

TTC = new\_dist / approach\_speed\_ms (if approaching > 0.5 m/s)

future\_cx = cx + (vx \* 1.0 s)

future\_cy = cy + (vy \* 1.0 s)

is\_cut\_in = (middle\_band AND abs(vx) > abs(vy)\*1.5 AND abs(vx) > THETA\*20)

will\_hit = PointInPolygon(future\_cx, future\_cy, Ego\_Lane\_Poly)

Stage1 = (new\_dist < 8.0 m)

Stage2 = (new\_dist <= 60.0 m AND TTC < 2.5 s)

IF will\_hit AND (approaching OR is\_cut\_in) AND (Stage1 OR Stage2):

TRIGGER ALERT("COLLISION IMMINENT")

END PROCEDURE

***Monocular Distance Estimation***

Distance is estimated using the Pinhole Camera Model with a known-width reference table (Car: 2.0 m; Truck: 2.5 m; Bus: 3.0 m; Pedestrian: 0.5 m):

***Distance = (Focal Length × Real Width) / Pixel Width***

A two-stage alert threshold is applied: Stage 1 triggers below 8 m; Stage 2 triggers between 8 m and 60 m when TTC falls below 2.5 seconds. This allows time-critical warnings at highway following distances without LiDAR or RADAR sensors.

*Fig. 1. Vector-Based Trajectory Prediction. The velocity vector (vxy) indicates the target vehicle is moving laterally toward the Ego Lane, triggering a collision alert despite the current safe distance. The system prioritizes trajectory over raw distance.*

*Fig. 2. Real-time Monocular Distance Estimation. Multiple targets are simultaneously tracked with class-specific reference widths applied per object. A Lane Departure Warning (red line indicating a left drift) is concurrently active.*

**E. Driver Monitoring System (DMS) and Drowsiness Detection**

A secondary inward-facing camera pipeline actively monitors the driver’s state using the MediaPipe Face Mesh framework, tracking 468 three-dimensional facial landmarks on a dedicated background thread. The Eye Aspect Ratio (EAR) is computed from six landmarks per eye:

***EAR = (‖p₁ − p₅‖ + ‖p₂ − p₄‖) / (2 × ‖p₀ − p₃‖)***

where p₁, p₂, p₄, p₅ are vertical eyelid landmarks and p₀, p₃ are horizontal extremities. When the eyes are open, EAR remains approximately constant; closure during a blink or microsleep causes it to drop sharply toward zero. If the bilateral averaged EAR remains below the calibrated threshold of 0.22 for more than 1.5 consecutive seconds, the system registers a Drowsy State, overrides the standard UI with a high-priority flashing warning, and triggers an escalating auditory alarm.

**III. Extended Platform Development**

Beyond the core embedded implementation, two parallel development efforts have produced an Android mobile ADAS application and an AI training data acquisition pipeline. These components extend the system’s platform reach and establish a pathway for continuous model improvement through data-driven refinement.

**A. Android Mobile ADAS Application (Development Stage)**

A native Android application has been developed in Kotlin, targeting commodity Android smartphones as a zero-hardware deployment platform for the core collision warning and lane departure capabilities. This constitutes an architectural port of the Python pipeline and is currently in active development. Its detection accuracy is measurably lower than the Python baseline (aiovidout5.py) across all tested scenario types, and it should be considered a research prototype rather than a production-grade system. Quantitative benchmarking against the 113-scenario test dataset is ongoing.

***YOLOv8 Inference via TensorFlow Lite***

Object detection is performed using a YOLOv8 model exported to TFLite Float16 format (selfdriving.tflite). The YoloDetector class implements a three-stage hardware delegate hierarchy to maximize inference throughput on heterogeneous devices: (1) GPU delegate is attempted first; (2) NNAPI delegate on failure; (3) CPU fallback as final resort. A manual letterboxing pre-processing step—matching the Ultralytics Python convention (scale = min(640/srcW, 640/srcH), centered padding on a 640×640 black canvas)—was implemented to resolve an aspect-ratio mapping error that corrupted bounding box coordinates in earlier revisions. The inverse transformation maps model-space coordinates back to original camera resolution. Only physically relevant object classes (biker, car, pedestrian, truck) pass the post-processing filter, and Non-Maximum Suppression is applied at an IoU threshold of 0.70 matching the Ultralytics default.

***AdasPhysicsEngine (Kotlin Port)***

The vector-based collision physics is ported to Kotlin as the AdasPhysicsEngine singleton, faithfully reproducing the Python logic: historical trajectory storage, velocity vector computation, two-stage TTC thresholds, ego-lane polygon intersection via ray-casting, and lateral cut-in detection. When lane detection yields valid lane boundaries, the ego-lane polygon is dynamically rebuilt from the detected left and right lane lines, making collision zone estimation adaptive rather than fixed. GPS fusion mode scales TTC thresholds with the measured vehicle speed (from Bluetooth telemetry), providing more conservative warnings at highway speeds. A user-configurable TTC multiplier (1.0×, 1.5×, or 2.0×) allows sensitivity adjustment without recompilation.

***LaneDetector (Kotlin, OpenCV-Free)***

Lane detection is performed by a pure pixel-scanning linear regression algorithm, avoiding the OpenCV dependency. The detector scans a trapezoidal ROI in an HLS-approximated color space identifying white pixels (luminance > 155) and yellow pixels (R > 140, G > 100, B < 90), then applies least-squares linear regression on accumulated pixel coordinates to yield lane boundary slopes and intercepts. Results are smoothed with an exponential moving average (α = 0.80). This approach is computationally lightweight but substantially less robust than the Probabilistic Hough Transform used in the Python implementation, particularly on curved roads, low-contrast markings, and under adverse lighting conditions.

***Accuracy Comparison and Known Limitations***

The Android implementation exhibits lower detection accuracy than the Python baseline for several compounding reasons. First, TFLite Float16 quantization introduces numerical precision loss relative to full-precision PyTorch inference, reducing confidence on borderline detections. Second, the Android camera pipeline introduces higher and less predictable per-frame latency than the USB camera on Raspberry Pi, degrading velocity vector estimation for fast-moving objects. Third, the pixel-regression lane detector is less robust than the Hough-based pipeline, particularly under curved roads and complex lighting. Fourth, the Android platform lacks a hardware accelerator equivalent to the Hailo-8 NPU; mobile GPU delegates are substantially faster than CPU inference but cannot match a dedicated AI accelerator’s throughput. The Android application may nonetheless be suitable for low-speed urban driving scenarios where larger available TTC margins partially compensate for lower detection rates.

***Additional Application Features***

Beyond ADAS functionality, the Android application incorporates voice command recognition (wake word: “oret”, implemented with edit-distance fuzzy matching), a lane drift audio alert, a dashcam recording mode with UI overlay or raw video options, and an OTA update mechanism. A voice command interface was developed using Android’s SpeechRecognizer API supporting commands for navigation, ADAS activation, Bluetooth connectivity, and recording control. These features enhance operational usability during vehicle deployment.

**B. Simulation-in-the-Loop Enhancements**

The BeamNG.drive SITL framework has been extended with two significant improvements: a multi-mode autopilot architecture and tightly integrated drowsiness monitoring.

***Multi-Mode Autopilot Architecture***

The autopilot now supports three selectable operating modes to facilitate incremental testing and failure-mode analysis:

* **Mode 1 — Steering Only:** Lane-keep steering is active; YOLO-based radar and collision avoidance are disabled. Used to validate lane tracking in isolation without confounding braking behavior.
* **Mode 2 — Braking Only:** YOLO radar and collision avoidance are active; lane-keep steering is disabled. Used to validate FCW braking commands independently of steering errors.
* **Mode 3 — Full Autopilot:** All subsystems active concurrently: lane-keep steering, YOLO radar, and drowsiness monitoring. This represents the complete Level 2 assistance mode validated in SITL.

***Integrated Drowsiness Monitoring in SITL***

The DrowsinessMonitor module, running MediaPipe Face Mesh on a background thread, is integrated directly into the SITL autopilot loop. An escalating intervention protocol is tied to drowsiness duration: during the first second of detected eye closure, a warning tone is issued and intermittent braking is applied; between 1 and 3 seconds, repeated escalating tones and cyclic braking are active; beyond 3 seconds of continuous closure, continuous braking and a persistent 1800 Hz alarm are triggered. This graduated response mirrors real-world ADAS intervention strategies and was validated in simulation prior to integration with the physical prototype.

**C. AI Training Data Acquisition Pipeline**

The aiovidout5.py tool introduces a human-in-the-loop training data pipeline designed to produce labeled datasets for future model fine-tuning and reinforcement learning (RL) experimentation. The pipeline operates in two sequential phases.

***Phase 1: Automated Collision Event Extraction***

During video processing, every instance in which a tracked vehicle reaches RED alert status is automatically logged. For each event, 17 features are recorded to a CSV file (crash\_training\_data\_v2.csv): track identifier, object class, bounding box centroid and dimensions, scene density (number of concurrent detections), ego-vehicle drift status, estimated distance (m), approach speed (km/h), lateral velocity (vx), vertical velocity (vy), Time-to-Collision (TTC, s), and contextual metadata (video source, output file path, timestamp). A JSON event log (events\_log.json) stores raw event data for the subsequent grading phase. A file-audit and resume checkpoint system allows processing to restart from the last saved frame after interruption, essential for large video datasets. The pipeline additionally supports YouTube video input via yt-dlp and an Intel OpenVINO model backend for accelerated processing on Intel hardware.

***Phase 2: Human-in-the-Loop Grading***

In the evaluation phase, reviewers assess each logged collision event through an FFmpeg-assisted video player that renders a frame-perfect 6-second clip (3 seconds before and after the event timestamp) for each incident. Reviewers grade three dimensions:

* **Severity:** (1) Actual crash; (2) Close call; (3) False alarm.
* **Warning Timing:** (a) Too late; (b) Too early; (c) Perfect timing.
* **Warning End Quality:** (1) On-time clearance; (2) Warning dropped prematurely.

A reward score (0–100) is computed from these grades: base scores of 100 (crash) and 80 (close call) are penalized for late timing (−30) and premature warning end (−20). False alarms receive a reward of zero. This scored dataset is intended to train a lightweight reinforcement learning agent or fine-tune threat classification thresholds beyond the hand-tuned values currently in use.

**IV. Integrated System Operation**

**A. Collision Warning Response**

The integrated FCW system proved highly effective in standard highway following scenarios. When the vector logic detects a trajectory intersecting the Ego Lane polygon, the visual interface transitions dramatically: a “COLLISION IMMINENT” overlay appears, the threatening vehicle’s bounding box changes from green to red, and the screen border flashes red. Objects within 30 meters and closing receive a reinforced Stage 1 alert. The cut-in logic successfully identified merging vehicles approximately 500 milliseconds before they fully crossed the lane markings, providing a critical reaction time advantage.

*Fig. 3. Temporal Progression of the Forward Collision Warning (FCW). (Top) Vehicle detected at safe following distance. (Middle) Target decelerates; trajectory analysis flags a potential impact. (Bottom) Visual “COLLISION IMMINENT” alert triggers when TTC drops below the safety threshold.*

**B. Real-Time Performance**

On the target Raspberry Pi 5 with Hailo-8 AI acceleration, the combined lane detection and YOLO inference pipelines consistently achieve 30+ frames per second at 1280×720 resolution in FORWARD\_MODE, well above the threshold for actionable driver alerts. State transition latency from reverse gear signal to rear-camera display is under 500 milliseconds. The auditory parking sensor feedback is immediately responsive, with increasing beep tempo providing clear urgency cues proportional to decreasing obstacle distance.

**C. Traffic Sign Recognition and Intelligent Speed Adaptation**

Leveraging the YOLOv8 pipeline, a dual-mode Intelligent Speed Adaptation (ISA) module processes detected speed limit signs. In physical retrofit deployments, the system operates as a passive advisory aid, comparing detected speed limits against vehicle speed and issuing audiovisual overspeeding alerts, as a low-cost retrofit lacks powertrain connectivity. In SITL testing, the ISA operates as a closed-loop active control feature: upon detecting a speed limit sign, the control algorithm overwrites the virtual vehicle’s maximum cruise speed, enabling autonomous deceleration without human input. This validates the control logic for future drive-by-wire hardware integration.

**D. Debug Mode and Development Tooling**

A --debug-mode command-line flag decouples the software from physical hardware, enabling development and tuning on standard PCs with simulated GPIO signals and keyboard-toggled operational state. A comprehensive debug overlay renders all processed data—lane lines, ROIs, bounding boxes, velocity vectors, distance labels, and status messages—on the live video feed, serving as an indispensable calibration and validation tool.

**V. Performance Evaluation and Test Results**

To validate system efficacy, a comprehensive analysis was conducted on a dataset of 113 crash and near-miss scenarios spanning diverse environmental conditions (daylight, nighttime, rain, dust) and accident typologies. Three evaluation metrics were applied: False Alarms (warnings issued when no threat existed), Perfect Warnings (timely alerts issued more than 1.5 seconds before impact), and Missed/Late Warnings (no alert, or alert issued fewer than 0.5 seconds before impact).

**A. Statistical Overview**

The system achieved a 0% False Alarm Rate (0 of 113 scenarios), validating the effectiveness of the vector-based filtering logic. By requiring a future trajectory intersection with the Ego Lane polygon, the system correctly ignores vehicles in adjacent lanes not on a collision course. Overall, timely warnings were generated in 60.1% (68 of 113) of total test cases, while the system failed to warn or warned too late in 39.8% (45 of 113) of cases. Significant performance variation across scenario types is documented in Table II.

**B. Scenario-Specific Performance**

**TABLE II: Scenario-Specific Detection Accuracy (113 Total Scenarios)**

| **Scenario Type** | **N** | **Successful Warnings (>1.5 s)** | **Missed / Late (<0.5 s)** | **Accuracy** |
| --- | --- | --- | --- | --- |
| **Longitudinal (Rear-End)** | 50 | 47 | 3 | **94%** |
| **Lateral (T-Bone / Intersection)** | 29 | 14 | 15 | **48%** |
| **Lane Cut-Ins (Partial Object)** | 22 | 6 | 16 | **27%** |
| **Nighttime / Low-Light** | 12 | 1 | 11 | **8%** |
| **TOTAL** | **113** | **68** | **45** | **60.1%** |

**C. Simulation-in-the-Loop (SITL) and Active Intervention Validation**

The system’s capacity for Level 2 active autonomous intervention was validated via SITL. The ADAS pipeline was bridged with BeamNG.drive through a Python telemetry node capturing the simulator’s visual output via high-speed screen capture while receiving physical vehicle telemetry (speed in km/h) via a local UDP socket on port 4444. The classical lane detection algorithm dynamically calculated road curvature and executed proportional steering corrections; the FCW module scaled TTC thresholds with the received speed telemetry. Upon detecting a RED collision threat or a severe drowsiness event (sustained eye closure > 1.5 s), the system successfully executed Automated Emergency Braking (AEB), safely halting the virtual vehicle. The three-mode autopilot architecture (Steering Only, Braking Only, Full AP) enabled independent validation of each subsystem before full integration testing, confirming that the underlying control logic is robust enough to actuate physical drive-by-wire systems in future hardware iterations.

**VI. Discussion**

**A. Partial Object Geometric Limitation (Cut-Ins)**

Testing revealed only a 27% detection rate for lane cut-ins. A fundamental limitation is inherent in the monocular distance formula: when a merging vehicle presents only a visible corner to the camera, YOLO generates a small bounding box around the fragment. The distance algorithm interprets this small box as a distant vehicle and computes a safe distance even when the object is physically adjacent to the bumper. This confirms that bounding-box geometry alone is insufficient for close-proximity cut-in detection without dedicated partial-object training data.

*Fig. 4. Lateral Threat Detection (T-Bone Logic). The system identifies a vehicle crossing the intersection perpendicular to the host vehicle; lateral velocity logic triggers a warning before the object enters the direct path of travel.*

*Fig. 5. False Positive Analysis: Adjacent Lane Interference. A vehicle in the adjacent lane drifts close to the lane marker; perspective distortion of the monocular camera causes the system to misclassify the vehicle as entering the Ego Lane.*

**B. Optical Expansion vs. Lateral Velocity**

The disparity between rear-end detection (94%) and T-bone detection (48%) highlights a fundamental challenge in monocular vision. Rear-end collisions are Z-axis events where the target scales symmetrically in the image plane. T-bone collisions are X-axis translations producing little or no scale change. Although lateral velocity (vx) is tracked, the 30 FPS frame rate combined with high cross-traffic speeds can result in the target traversing the Ego Lane polygon in fewer than three frames—too rapidly for the velocity smoothing algorithm to confirm a collision vector with sufficient confidence.

**C. Environmental and Sensor Constraints**

The 8% detection rate in pitch-dark conditions confirms that visible-light cameras alone are insufficient for 24-hour operational coverage. At night, object detection confidence frequently drops below the 0.4 threshold, causing the system to discard valid targets as noise. Integration of near-infrared illumination or thermal imaging is essential for nighttime deployment.

**D. The System as an Aid, Not an Autopilot**

It is imperative to characterize this system accurately: it is a driver assistance tool and not a replacement for an attentive human driver. The YOLO model may produce false negatives or misclassifications; the lane algorithm may be momentarily confused by shadows, pavement changes, or road works. All alerts are designed to recapture a distracted driver’s attention, not to make driving decisions. Ultimate responsibility for safe vehicle operation remains entirely with the driver.

**E. Android Platform Accuracy Gap**

In addition to TFLite Float16 quantization loss, Android camera pipeline latency, and the absence of NPU-equivalent acceleration (discussed in Section III.A), a fourth degradation factor is the LaneDetector’s pixel regression approach, which is less stable than the Hough-based pipeline under real-world road conditions. The combined effect produces a detection accuracy gap relative to the embedded Python baseline that is practically significant in high-speed scenarios. The Android platform remains potentially useful for urban driving at speeds below 60 km/h, where higher TTC margins provide more time for late detections to still be actionable. Future work will quantify each degradation factor independently to prioritize the highest-impact improvements.

**F. Hybrid AI and Vector Physics Approach**

A central finding is the necessity of combining deep learning with deterministic physics. YOLO detects presence, not trajectory. By adding the vector physics layer, the system eliminates false positives caused by vehicles traveling in parallel adjacent lanes: a co-moving vehicle is correctly classified as safe because its future trajectory does not intersect the Ego Lane. Conversely, an aggressively merging vehicle is immediately flagged because its lateral velocity vector does intersect. This hybrid approach provides substantially greater safety reliability than pure area-based bounding-box methods.

**G. Component Reliability and Automotive Durability**

Automotive electronics are typically certified to AEC-Q100 standards, designed for temperature ranges of −40°C to +125°C, continuous vibration, and humidity. The Raspberry Pi and USB cameras use friction-fit connectors susceptible to vibration-induced displacement. The MicroSD storage medium risks data corruption during sudden power loss and has limited write-cycle endurance compared to automotive eMMC storage. Consumer-grade JSN-SR04T sensors may require replacement every 2–3 years of continuous operation, versus the longer service life of OEM ultrasonic sensors.

**H. Headlight Control and Thermal Management**

The relay-based headlight retrofit performs binary On/Off switching only; modern vehicles using PWM-based dimming or CAN bus–managed lighting may require load resistors to prevent “bulb out” warnings. The combined thermal output of the Cortex-A76 CPU and Hailo-8 NPU requires active cooling; a PID-controlled fan and a regulated 5V/5A buck converter with input capacitance are recommended to maintain stability under estimated peak load of approximately 12 W in high-ambient-temperature automotive environments.

**VII. Future Work and Potential Improvements**

The current implementation establishes a robust and extensible platform for several high-impact future enhancements:

* **Deep Learning Lane Detection:** Replacing the Hough-based pipeline with a lightweight semantic segmentation model (e.g., ENet or MobileNetV2-based architecture) would substantially improve robustness to shadows, weather, and atypical road markings by learning texture and context rather than relying on geometric properties.
* **IMU Sensor Fusion:** Integrating an Inertial Measurement Unit and fusing its yaw rate and acceleration data with camera input via a Kalman filter would enable trajectory estimation during brief lane-line occlusion intervals, preventing unnecessary warnings in curves.
* **Stereoscopic Distance Validation:** A secondary forward-facing camera would enable triangulation-based distance measurement, eliminating dependence on assumed object widths and improving accuracy for non-standard vehicle sizes.
* **Partial Object Training for Cut-Ins:** Training the detection model to classify vehicle corners as distinct classes (e.g., “Rear Left Corner”) would allow the distance algorithm to switch to a close-proximity mode when only a vehicle fragment is visible.
* **Night Vision Sensor Fusion:** Integration of a low-cost microbolometer thermal camera or IR-illuminated sensor would resolve the 92% nighttime failure rate by detecting thermal signatures of vehicles and pedestrians independently of ambient lighting.
* **Android Application Maturation:** Future work will benchmark the Android implementation against the 113-scenario test dataset, quantify each per-factor accuracy degradation, and explore hardware AI delegates to close the gap with the embedded baseline.
* **RL-Based Threat Classifier:** The reward-scored dataset produced by the aiovidout5.py pipeline is intended to train a lightweight reinforcement learning agent to optimize threat classification thresholds beyond the current hand-tuned values.
* **LiDAR–Camera Sensor Fusion for AEB:** Projecting Velodyne VLP-16 3D point cloud data onto the 2D camera plane via extrinsic calibration matrices would provide millimeter-accurate depth for AEB, eliminating the monocular width-assumption dependency.
* **Automated Camera Calibration:** A user-guided on-screen calibration routine aligning reference points with the road vanishing point would automatically compute the correct perspective transform for each vehicle installation.
* **Hill Start Assist (HSA) for Custom Drive-by-Wire Platforms:** An IMU-driven HSA feature is proposed for custom drive-by-wire experimental vehicles, using BNO055 pitch angle to autonomously apply braking on inclines exceeding 5 degrees. This is explicitly not recommended for commercial vehicles, where retrofit actuation of hydraulic brake systems introduces ISO 26262 and legal liability risks.

**VIII. Cost Analysis and Economic Feasibility**

A primary motivation of this research is to democratize safety technology by providing a cost-effective alternative to proprietary systems. A detailed Bill of Materials (BOM) was compiled based on current single-unit market prices.

**A. Bill of Materials (Prototype)**

**TABLE I: Bill of Materials — Prototype ADAS Implementation**

| **Component** | **Specification** | **Est. Cost (EUR)** |
| --- | --- | --- |
| Compute Module | Raspberry Pi 5 (16 GB RAM) + Active Cooler | 145 |
| AI Acceleration | Hailo-8 M.2 Module (26 TOPS) + M.2 HAT Adapter | 140 |
| Visual Sensors | 2× USB 1080p Wide-Angle Cameras + 1× USB 720p Rear Camera | 100 |
| Display | 7-inch Capacitive Touchscreen (DSI/HDMI) | 65 |
| Proximity Sensors | 6× JSN-SR04T Waterproof Ultrasonic Sensors | 45 |
| Power Management | 12V-to-5V 5A Buck Converter + Relay Module | 25 |
| Peripherals | ADC (MCP3008), Wiring, Enclosure, PCB | 30 |
| **TOTAL** |  | **545** |

**B. Installation and Total Cost of Ownership**

DIY installation (basic mechanical skills: trim removal, cable routing, wire tapping) adds EUR 0–20 in consumables. Professional installation by an automotive electrician, covering bumper drilling for ultrasonic sensors, rear camera chassis routing, and power relay wiring, requires 3–5 hours at approximately EUR 200 at average European labor rates.

**C. Market Comparison and Scalability**

Current aftermarket ADAS solutions from vendors such as Mobileye or high-end Garmin units range from EUR 800 to EUR 1,600, typically excluding mandatory professional installation. Even in the maximum-cost scenario (prototype hardware + professional installation), the proposed system totals approximately EUR 745—a 30–60% saving compared to commercial equivalents—while integrating parking assistance, blind-spot monitoring, headlight automation, lane departure warning, and drowsiness detection in a single unit.

**IX. Regulatory, Safety, and Ethical Implications**

Technical feasibility is a necessary but insufficient condition for commercial deployment. Productizing this prototype as a consumer retrofit requires addressing the automotive industry’s rigorous safety and regulatory frameworks.

**A. Functional Safety (ISO 26262)**

ISO 26262 defines functional safety as the absence of unreasonable risk from malfunctioning electrical/electronic systems. The consumer-grade Raspberry Pi 5 lacks automotive environmental certification (temperature: −40°C to +125°C; continuous vibration; humidity). The current software architecture does not satisfy the redundancy requirements for higher ASIL levels. ASIL-D certification (required for safety-critical functions such as steering and braking) demands redundant calculation paths and lockstep processing that a single-board computer architecture cannot provide.

**B. Safety of the Intended Functionality (SOTIF — ISO 21448)**

SOTIF addresses hazards from AI functional insufficiencies. Without CAN bus integration, the system cannot detect the driver’s turn signal, making it unable to distinguish an intentional lane change from an unintentional drift, resulting in false alarms during planned maneuvers. A production implementation would require integration with the vehicle’s data network to access turn indicator, speed, and steering angle data.

**C. Legal Liability and Homologation**

Commercial deployment faces ambiguous liability attribution in retrofit scenarios—uncertainty as to whether fault lies with the software developer, installer, or driver. Installation of aftermarket screens must comply with local homologation standards (e.g., UN ECE R46 for indirect vision; FMVSS 111 in the US). Rear-view camera systems introducing latency exceeding 200 milliseconds would fail regulatory compliance checks.

**D. The Human-in-the-Loop Problem**

Automation complacency—the tendency to reduce attentiveness in the presence of an assistance system—represents a recognized safety risk. Without the rigorous Human-Machine Interface testing conducted by major automakers, an overly aggressive warning system may induce alarm fatigue, causing drivers to disregard all alerts, or may lead to over-reliance at the expense of active visual scanning. Calibration of alert aggressiveness and clear communication of system limitations to the driver are essential for safe deployment.

**X. Conclusion**

This research has successfully demonstrated the design, implementation, and integration of a functional, low-cost, multi-modal Advanced Driver-Assistance System on an embedded platform. By synergistically combining a classical vision-based Lane Departure Warning system, a deep learning–based object detector, a vector-physics Forward Collision Warning module, an automated ultrasonic reverse assist, and a facial landmark drowsiness monitor, the project provides a holistic functional prototype for dramatically enhancing safety in vehicles lacking modern ADAS features. The system’s multi-threaded, state-driven architecture is efficient; its embedded performance is suitable for real-time application; and its open-source foundation makes it a replicable blueprint for democratizing automotive safety technology.

Parallel development efforts have extended the system’s reach: a native Android ADAS application ports the core collision warning pipeline to commodity smartphones and is actively under development, currently exhibiting lower detection accuracy than the embedded baseline and targeting a maturation roadmap that includes quantitative benchmarking and hardware delegate optimization. An AI training data acquisition pipeline with human-in-the-loop grading and reward scoring establishes a systematic pathway for data-driven threshold optimization and future reinforcement learning integration.

The clear path forward lies in leveraging more advanced perception models, integrating sensor fusion for nighttime and adverse-weather robustness, maturing the Android platform to close the accuracy gap with the embedded baseline, and progressing toward ISO 26262 compliance for eventual productization. Ultimately, this work affirms the substantial potential of embedded and mobile vision systems to bring life-saving safety technology out of the premium automotive segment and make it accessible to all drivers.

**References**

[1] J. Redmon and A. Farhadi, “YOLOv3: An Incremental Improvement,” arXiv preprint arXiv:1804.02767, 2018.

[2] International Organization for Standardization, “Road vehicles — Functional safety — Part 1: Vocabulary,” ISO 26262-1:2018, 2018.

[3] G. Bradski, “The OpenCV Library,” Dr. Dobb’s Journal of Software Tools, 2000.

[4] International Organization for Standardization, “Road vehicles — Safety of the intended functionality,” ISO 21448:2022, 2022.

[5] Raspberry Pi Foundation, “Raspberry Pi 5 Product Brief,” Nov. 2023. [Online]. Available: https://datasheets.raspberrypi.com/rpi5/raspberry-pi-5-product-brief.pdf

[6] Hailo, “Hailo-8 AI Processor Datasheet,” 2023. [Online]. Available: https://hailo.ai/products/ai-accelerators/hailo-8-ai-processor-for-edge-devices/

[7] Ultralytics, “YOLOv8: Real-Time Object Detection and Segmentation,” 2023. [Online]. Available: https://github.com/ultralytics/ultralytics

[8] Google LLC, “MediaPipe Solutions Guide,” 2023. [Online]. Available: https://developers.google.com/mediapipe

[9] T. Soukupová and J. Čech, “Real-Time Eye Blink Detection Using Facial Landmarks,” in Proc. 21st Computer Vision Winter Workshop (CVWW), Rimske Toplice, Slovenia, 2016.

[10] Google LLC, “TensorFlow Lite for On-Device ML,” 2023. [Online]. Available: https://www.tensorflow.org/lite

[11] BeamNG GmbH, “BeamNG.drive Vehicle Simulation Platform,” 2023. [Online]. Available: https://www.beamng.com

[12] Intel Corporation, “OpenVINO Toolkit Documentation,” 2023. [Online]. Available: https://docs.openvino.ai