# DPU Accelerated Inference (KV260 FPGA DPUCZDX8G)

## Purpose
This is the **hardware-accelerated** stage of the project: run CNN inference on the **FPGA-based DPU (DPUCZDX8G)** and compare it to the CPU baseline (**1.37 FPS**).

Requirement coverage:
- **≥ 2× speedup vs CPU-only** → achieved **19.1× to 86.4×**
- Real-time / near real-time object detection on embedded hardware

---

## What `dpu_inference.py` Does (3 Parts)

1. **Benchmark (throughput + latency)**
   - Runs Vitis-AI `test_performance_*` on multiple models
   - Tests **1-thread** and **4-thread**
   - Captures **FPS**, **E2E latency**, and (when available) **DPU-only latency**

2. **Run detection on images**
   - Uses Vitis-AI `test_jpeg_ssd` on images in `test_images/`
   - Parses detections (class / box / confidence)
   - Saves annotated images

3. **Generate a final report**
   - Compares CPU baseline vs DPU results
   - Computes speedup per model and thread count
   - Writes **TXT / JSON / CSV**

---

## How It Fits in the Pipeline

```text
Step 1: CPU baseline (OpenCV DNN, FP32)  ->  ~1.37 FPS
Step 2: Quantize + compile (FP32 -> INT8 -> xmodel)
Step 3: DPU inference (this)            ->  26.1–118 FPS
```

---

## HW/SW Co‑Design (What runs where)

- **ARM CPU (PS):** camera I/O, preprocess (resize/mean), postprocess (decode/NMS), drawing/saving
- **FPGA DPU (PL):** **CNN forward pass** (INT8) @ ~300 MHz, massively parallel MACs

Why speedup happens:
- CPU inference: **~716 ms**
- DPU inference: **~16 ms** (≈ **44×** faster for the forward pass)
- End-to-end also includes pre/post + transfer overhead → **~38 ms** total (≈ **19×**)

---

## DPU Details (KV260)
| Property | Value |
|---|---|
| DPU IP | DPUCZDX8G |
| Freq | 300 MHz |
| Precision | INT8 |
| Peak | ~1.2 TOPS |
| Variant | B4096 (4096 MACs/cycle class) |

---

## Models Benchmarked
- **SSD MobileNet V2** (primary apples-to-apples baseline comparison)
- **SSD ADAS Pruned 0.95** (very fast detection model)
- **ResNet50** (classification reference)

---

## Run Instructions (KV260)


### 1) Prepare test images
```bash
cd dpu_inference
mkdir -p test_images
# copy .jpg images into test_images/
```

### 2) Run everything
```bash
python3 dpu_inference.py
```

### Manual benchmark examples (optional)
```bash
# SSD benchmark
cd /home/root/Vitis-AI/demo/Vitis-AI-Library/samples/ssd
./test_performance_ssd ssd_mobilenet_v2 test_performance_ssd.list -t 1 -s 30

# Classification benchmark
cd /home/root/Vitis-AI/demo/Vitis-AI-Library/samples/classification
./test_performance_classification resnet50 test_performance_classification.list -t 1 -s 30
```

---

## Results (Example Summary)

```text
CPU baseline reference: 1.37 FPS (728.73 ms E2E)

ssd_mobilenet_v2 (1 thread):  26.13 FPS, 38.23 ms E2E, 16.11 ms DPU  -> 19.1× E2E speedup
ssd_mobilenet_v2 (4 thread):  61.45 FPS                              -> 44.9× throughput speedup
ssd_adas_pruned_0_95 (4 thr): 118.40 FPS                              -> 86.4× throughput speedup
```

Notes:
- “Pure inference” speedup is larger than E2E speedup because **pre/post + DPU I/O overhead** still runs on CPU.

---

## Output Files / Folder Layout

```text
4_dpu_inference/
├── dpu_inference.py
├── README.md
├── test_images/                 # input .jpg files
└── results/
    ├── dpu_report.txt
    ├── dpu_results.json
    ├── benchmark_results.csv
    └── frames/                  # annotated output images
        ├── ssd_mobilenet_v2_*.jpg
        └── ssd_adas_pruned_0_95_*.jpg
```

---

## What This Proves
- The FPGA DPU accelerates the CNN forward pass enough to meet and exceed the project requirement.
- The comparison is measurable and repeatable (CSV/JSON/TXT artifacts).
- The system demonstrates real HW/SW partitioning: CPU handles I/O + light processing, DPU handles compute-heavy inference.
