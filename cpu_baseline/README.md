# CPU Baseline: Object Detection on ARM Cortex‑A53

## Purpose
This is the **software-only CPU baseline** for the project. Before any FPGA/DPU acceleration, we measure end-to-end CNN performance on the **ARM Cortex‑A53 CPU only**. This baseline is required to prove the requirement:

> **Minimum 2× speedup compared to software-only CNN execution on Arm CPU**

---

## What `cpu_baseline.py` Does
1. Opens a USB camera on the **KV260**
2. Captures **100 frames** at **1280×960**
3. For each frame:
   - **Preprocess:** resize to **300×300**, create OpenCV DNN blob  
   - **Infer (CPU):** run **SSD MobileNet V2** with OpenCV DNN  
   - **Postprocess:** decode detections, filter by confidence
4. Measures latency for each stage (pre / inference / post)
5. Draws bounding boxes and saves annotated frames
6. Exports a performance report (**TXT / JSON / CSV**)

**No FPGA, no DPU, no hardware acceleration.**

---

## How It Fits Into the Project
- **Step 1 (this):** CPU baseline (OpenCV DNN on ARM)
- **Step 2:** Quantize + compile (FP32 → INT8 → `xmodel`)
- **Step 3:** Run the same model on the DPU (FPGA) and compare speedup

Example outcome:
- CPU: ~**1.37 FPS**
- DPU: ~**26.1 FPS** (≈ **19×** speedup)

---

## Platform Details

| Component | Specification |
|---|---|
| Board | Kria KV260 Vision AI Starter Kit |
| SoC | Zynq UltraScale+ MPSoC (XCK26) |
| CPU | Quad-core ARM Cortex‑A53 @ 1.2 GHz |
| Memory | 4 GB DDR4 |
| OS | PetaLinux (Ubuntu-based) |
| Camera | USB camera (1280×960) |
| Framework | OpenCV 4.x (DNN module) |
| Acceleration | **None** |

---

## Model: SSD MobileNet V2 (COCO)

| Property | Value |
|---|---|
| Input size | 300 × 300 × 3 |
| Dataset | COCO (91 classes) |
| Format | TensorFlow frozen graph (`.pb`) |
| Precision | FP32 |
| Size | ~67 MB |
| Params | ~4.3M |

Why this model:
- Lightweight and common for embedded use
- Still **too slow on CPU**, motivating acceleration

---

## How to Run

### 1) Install dependencies (KV260)
```bash
pip3 install opencv-python numpy
```

### 2) Download model
```bash
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar -xzf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
```

Ensure the OpenCV config file is present:
- `ssd_mobilenet_v2_coco_2018_03_29.pbtxt` (place it next to the model / script as expected)

### 3) Run benchmark
```bash
cd arm-soc-challenge
python3 cpu_baseline.py
```

---

## Expected Output (Example)

Per-frame log:
```text
Frame 001 | Latency:  782.3ms | FPS:  1.3 | Objects: person(87%), chair(72%)
...
Frame 100 | Latency:  718.9ms | FPS:  1.4 | Objects: chair(65%)
```

Summary report:
```text
Average Latency:       728.73 +/- 38.45 ms
Average FPS:           1.37

Latency Breakdown:
  Preprocessing:       10.21 ms  ( 1.4%)
  Inference (CPU):     715.92 ms (98.2%)
  Postprocessing:      1.97 ms   ( 0.3%)
```

Key observations:
1. **Inference dominates** (~98% of time) → main bottleneck targeted by FPGA/DPU  
2. **~1.37 FPS** → far from real-time (≥30 FPS)  
3. Pre/post are small and acceptable to remain on CPU

---

## Output Files
After execution:
```text
cpu_results/
├── cpu_report.txt
├── cpu_results.json
├── cpu_latency.csv
└── frames/
    ├── frame_001.jpg
    ├── frame_002.jpg
    └── ...
```
