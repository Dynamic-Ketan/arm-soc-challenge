# FPGA-Accelerated Object Detection on Kria KV260

**Real-time CNN inference using HW/SW co-design on AMD-Xilinx Zynq UltraScale+ MPSoC**

[![Platform](https://img.shields.io/badge/Platform-Kria%20KV260-blue)]()
[![DPU](https://img.shields.io/badge/DPU-DPUCZDX8G%20B4096-green)]()
[![Speedup](https://img.shields.io/badge/Speedup-19×--86×%20vs%20CPU-red)]()

---

## Overview

This project demonstrates an end-to-end **hardware/software co-design** workflow for embedded object detection on the **Kria KV260 Vision AI Starter Kit** (Zynq UltraScale+ MPSoC). The same overall workload is measured in two modes:

1. **CPU baseline (software-only):** SSD MobileNet runs entirely on the **ARM Cortex‑A53** using OpenCV DNN (FP32).
2. **FPGA acceleration:** the CNN forward pass is offloaded to the **DPUCZDX8G** in FPGA fabric (PL), using an **INT8 quantized + compiled** model (`.xmodel`) executed via Vitis AI.

The goal is not just “it runs”, but to produce **measurable evidence** (FPS + latency + reports) that FPGA acceleration meets the requirement:

> **Requirement:** ≥ **2×** speedup vs software-only CNN on ARM CPU  
> **Achieved:** **~19×–86×** depending on model and threading.

### Headline Numbers (SSD MobileNet V2)

| Metric | CPU-Only (ARM A53) | DPU (FPGA) | Speedup |
|---|---:|---:|---:|
| **FPS (1 thread)** | 1.37 | 26.13 | **19.1×** |
| **FPS (4 threads)** | — | 61.45 | **44.9×** |
| **E2E latency** | 728.73 ms | 38.23 ms | **19.1×** |
| **Inference-only latency** | 715.92 ms | 16.11 ms | **44.4×** |

**Important detail:** inference-only speedup is larger than end-to-end speedup because **pre/post-processing and some transfer overhead remain on the CPU**. This is expected and is part of the HW/SW partitioning.

---

## Architecture (HW/SW Co-Design)

At a system level, KV260 is split into two domains:

- **PS (Processing System / ARM CPU):** camera I/O, preprocessing, postprocessing, control code
- **PL (Programmable Logic / FPGA):** DPU accelerator for the CNN forward pass

```
┌──────────────────────────────────���──────────────────────────────┐
│                        Kria KV260 SoC                           │
│                                                                 │
│  ┌──────────────────────┐      ┌──────────────────────────────┐ │
│  │   Processing System  │      │    Programmable Logic (PL)   │ │
│  │      (PS — CPU)      │      │                              │ │
│  │                      │      │  ┌────────────────────────┐  │ │
│  │  ARM Cortex-A53 ×4   │      │  │   DPUCZDX8G (B4096)     │  │ │
│  │  @ 1.2 GHz           │◄────►│  │   INT8 · 300 MHz        │  │ │
│  │                      │ AXI  │  │   ~1.2 TOPS             │  │ │
│  │  • Camera capture    │      │  │                        │  │ │
│  │  • Preprocessing     │      │  │   CNN forward pass      │  │ │
│  │  • Postprocessing    │      │  │   (quantized model)     │  │ │
│  │  • Display / save    │      │  └────────────────────────┘  │ │
│  └──────────┬───────────┘      └──────────────────────────────┘ │
│             │                                                   │
│        ┌────┴────┐                                              │
│        │ DDR4 4GB│                                              │
│        └─────────┘                                              │
└─────────────────────────────────────────────────────────────────┘
         │
    ┌────┴────┐
    │USB Cam  │  1280×960
    └─────────┘
```

### What runs where (practical view)

| Stage | Runs on | Why |
|---|---|---|
| Camera capture | CPU | device drivers + I/O |
| Preprocess (resize/normalize) | CPU | simple pixel ops; small relative cost |
| **CNN forward pass** | **DPU (FPGA)** | dominates CPU time; massively parallel on DPU |
| Postprocess (decode + NMS) | CPU | control-heavy, lightweight |
| Visualization / save frames | CPU | OpenCV convenience |

On CPU-only execution, **~98%** of the frame time is the CNN forward pass; moving only that part to FPGA yields the largest return.

---

## Project Structure (Repo Map)

This repository is organized as a 3-step pipeline so that results are reproducible and easy to validate:

```
.
├── README.md                          ← You are here
│
├── 2_cpu_baseline/                    ← Step 1: software-only baseline
│   ├── cpu_baseline.py                ← OpenCV DNN (FP32) on ARM CPU
│   ├── README.md
│   ├── ssd_mobilenet_v2_coco_2018_03_29/
│   │   ├── frozen_inference_graph.pb  ← FP32 TensorFlow frozen graph
│   │   └── ...
│   ├── ssd_mobilenet_v2_coco_2018_03_29.pbtxt
│   └── cpu_results/                   ← generated outputs (report/CSV/frames)
│
├── 3_model_conversion/                ← Step 2: quantize + compile to xmodel
│   ├── README.md
│   ├── compiled_model/                ← output .xmodel + metadata
│   └── logs/                          ← compiler/quantization logs
│
└── 4_dpu_inference/                   ← Step 3: accelerated inference on DPU
    ├── dpu_inference.py               ← benchmark + detection + report
    ├── test_images/                   ← input JPEGs for detection tests
    └── results/                       ← generated outputs (report/CSV/frames)
```

If you are evaluating the project:
- For baseline evidence, look at `cpu_baseline/cpu_results/`
- For acceleration evidence, look at `4pu_inference/results/`
- For conversion provenance, look at `model_conversion/`

---

## Results (Expanded)

### 1) Throughput (FPS)

The DPU increases throughput from “non-real-time” to “near/real-time”, and multi-threading improves throughput further by overlapping CPU tasks and DPU work.

```
                        FPS (frames per second)
  CPU baseline         ▓▓ 1.37
  DPU 1-thread         ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 26.13          (19.1×)
  DPU 4-thread         ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 61.45  (44.9×)
  DPU ADAS 4-thr       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 118.40 (86.4×)
                  0        20        40        60        80       100       120
```

### 2) Latency (per frame)

Latency is the more meaningful metric for real-time responsiveness.

| Stage | CPU-Only | DPU-Accelerated |
|---|---:|---:|
| Preprocessing | 10.21 ms | ~10 ms |
| **Inference** | **715.92 ms** | **16.11 ms** |
| Postprocessing | 1.97 ms | ~2 ms |
| **End-to-End** | **728.73 ms** | **38.23 ms** |

**Interpretation:**
- The DPU turns the CNN from **~0.7 seconds** into **~0.016 seconds**.
- End-to-end remains higher because the system still pays for:
  - CPU preprocessing/postprocessing
  - transfers / scheduling overhead between PS and PL

This is exactly what HW/SW co-design implies: move the *dominant compute kernel* to hardware.

### 3) Multi-model benchmarks

Different models stress the system differently:
- SSD MobileNet V2 is the baseline comparison model
- SSD ADAS pruned is optimized for higher FPS
- ResNet50 is a classification reference point

| Model | Threads | FPS | Notes |
|---|---:|---:|---|
| SSD MobileNet V2 | 1 | 26.13 | primary apples-to-apples comparison |
| SSD MobileNet V2 | 4 | 61.45 | throughput boosted via parallelism |
| SSD ADAS Pruned 0.95 | 4 | 118.40 | highest throughput in tested set |
| ResNet50 | 1 | ~70+ | classification reference |

(Exact numbers may vary slightly depending on overlay, clocks, I/O, and image sizes.)

---

## Reproducing the Results (Step-by-step)

### Prerequisites

| Item | Details |
|---|---|
| Board | Kria KV260 Vision AI Starter Kit |
| OS image | PetaLinux image with Vitis AI runtime (VART) |
| DPU overlay | e.g., `kv260-smartcam` providing DPUCZDX8G B4096 |
| Camera | USB UVC camera |
| Host PC | Docker + Vitis AI (for compilation step) |

---

### Step 1 — CPU baseline (run on KV260)

```bash
cd cpu_baseline
pip3 install opencv-python numpy

# Download the TF model (if not already present)
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar -xzf ssd_mobilenet_v2_coco_2018_03_29.tar.gz

python3 cpu_baseline.py
# outputs -> 2_cpu_baseline/cpu_results/
```

Outputs to inspect:
- `cpu_report.txt` (summary)
- `cpu_latency.csv` (per-frame breakdown)
- `frames/` (annotated evidence)

---

### Step 2 — Model conversion (run on host PC)

The DPU expects an **INT8 compiled model** (`.xmodel`). This step performs:
1) quantization (FP32 → INT8) and
2) compilation for the KV260 DPU arch.

```bash
# See model_conversion/README.md for the full procedure
# Example (illustrative):
vai_c_tensorflow \
  --frozen_pb quantize_eval_model.pb \
  --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
  --output_dir compiled_model \
  --net_name ssd_mobilenet_kv260
```

Expected artifact:
- `compiled_model/*.xmodel`

---

### Step 3 — DPU inference (run on KV260)


Run benchmarks + detection:
```bash
cd dpu_inference
mkdir -p test_images
# put some .jpg files in test_images/
python3 dpu_inference.py
# outputs -> 4_dpu_inference/results/
```

Outputs to inspect:
- `dpu_report.txt` (summary)
- `benchmark_results.csv` (plotting/analysis)
- `frames/` (annotated output)

---

## Platform Specifications

| Component | Specification |
|---|---|
| Board | Kria KV260 Vision AI Starter Kit |
| SoC | Zynq UltraScale+ MPSoC (XCK26) |
| CPU | Quad-core ARM Cortex‑A53 @ 1.2 GHz |
| FPGA | Programmable Logic fabric |
| DPU | DPUCZDX8G B4096, 300 MHz, INT8 |
| Peak perf | ~1.2 TOPS (INT8) |
| Memory | 4 GB DDR4 |
| Camera | USB UVC, 1280×960 |

---

## Key Takeaways (Why the speedup is real)

1. **CPU inference is the bottleneck**  
   The CNN forward pass takes ~716 ms/frame on ARM A53, dominating total time.

2. **DPU acceleration targets exactly that bottleneck**  
   The DPU runs the quantized CNN in ~16 ms (inference-only), cutting the largest cost.

3. **End-to-end speedup is lower than inference-only speedup (and that’s expected)**  
   Pre/post and transfer overhead remain; HW/SW co-design doesn’t eliminate them, it minimizes the dominant term.

4. **Threading improves throughput**  
   With multiple threads, CPU-side work and DPU execution can overlap better, increasing overall FPS.

---

## License / Attribution
- Model weights and configs originate from:
  - TensorFlow Model Zoo
  - Xilinx/AMD Vitis AI Model Zoo
- Each retains its respective license terms.
