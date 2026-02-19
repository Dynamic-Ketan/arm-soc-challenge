#!/usr/bin/env python3
"""
DPU Accelerated Object Detection - Final Version
Platform: Kria KV260 (Zynq UltraScale+ MPSoC)
DPU: DPUCZDX8G @ 300MHz
Framework: Vitis-AI 1.4

Part 1: Pure DPU benchmarks (multiple models, threads)
Part 2: DPU detection on test images with bounding boxes
Part 3: Complete performance report with CPU comparison

Note: Vitis-AI binaries and xmodels are pre-installed on the
      KV260 SD card image at standard system paths.
"""

import cv2
import numpy as np
import subprocess
import time
import os
import json
from collections import Counter
from datetime import datetime

# ============================================================
# Configuration
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
TEST_IMAGES_DIR = os.path.join(SCRIPT_DIR, "test_images")
BENCHMARK_DURATION = 30
CONF_THRESHOLD = 0.3

# Vitis-AI system paths (pre-installed on KV260 SD image)
SSD_DIR = "/home/root/Vitis-AI/demo/Vitis-AI-Library/samples/ssd"
CLS_DIR = "/home/root/Vitis-AI/demo/Vitis-AI-Library/samples/classification"
SSD_JPEG_BINARY = os.path.join(SSD_DIR, "test_jpeg_ssd")
SSD_PERF_BINARY = os.path.join(SSD_DIR, "test_performance_ssd")
CLS_PERF_BINARY = os.path.join(CLS_DIR, "test_performance_classification")

# CPU baseline results (from cpu_baseline.py measurement)
CPU_BASELINE = {
    "model": "SSD MobileNet V2 (COCO)",
    "platform": "ARM Cortex-A53 @ 1.2 GHz",
    "input": "Live Camera (1280x960)",
    "num_frames": 100,
    "avg_fps": 1.37,
    "avg_latency": 728.73,
    "avg_inference": 715.92,
    "avg_preprocess": 10.21,
    "avg_postprocess": 1.97,
    "objects": {"person": 120, "chair": 112}
}

# Colors for bounding boxes
COLORS = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 255, 0), (255, 128, 0)
]


# ============================================================
# Part 1: DPU Performance Benchmarks
# ============================================================
def run_benchmark(binary_dir, binary_name, model_name, list_file, threads, duration):
    """Run Vitis-AI performance benchmark from correct directory"""
    original_dir = os.getcwd()
    os.chdir(binary_dir)

    cmd = [
        "./" + binary_name, model_name, list_file,
        "-t", str(threads), "-s", str(duration)
    ]

    result_data = {
        "model": model_name,
        "threads": threads,
        "duration_s": duration,
        "fps": 0,
        "e2e_mean_ms": 0,
        "dpu_mean_ms": 0,
        "status": "failed"
    }

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 30)
        output = result.stdout + result.stderr

        for line in output.split('\n'):
            if 'FPS=' in line:
                try:
                    result_data["fps"] = float(line.split('FPS=')[1].split()[0])
                except:
                    pass
            if 'E2E_MEAN=' in line:
                try:
                    val = float(line.split('E2E_MEAN=')[1].split()[0])
                    result_data["e2e_mean_ms"] = round(val / 1000.0, 2)
                except:
                    pass
            if 'DPU_MEAN=' in line:
                try:
                    val = float(line.split('DPU_MEAN=')[1].split()[0])
                    result_data["dpu_mean_ms"] = round(val / 1000.0, 2)
                except:
                    pass

        if result_data["fps"] > 0:
            result_data["status"] = "success"

    except Exception as e:
        print("    ERROR: {}".format(e))

    os.chdir(original_dir)
    return result_data


def run_all_benchmarks():
    """Run complete DPU benchmark suite"""
    print("\n" + "=" * 60)
    print("PART 1: DPU PERFORMANCE BENCHMARKS")
    print("Running on FPGA DPU (DPUCZDX8G @ 300MHz)")
    print("=" * 60)

    results = []

    # SSD Object Detection models
    ssd_models = [
        ("ssd_mobilenet_v2", "SSD MobileNet V2 (VOC)"),
        ("ssd_adas_pruned_0_95", "SSD ADAS Pruned 95%"),
    ]

    for model_name, description in ssd_models:
        for threads in [1, 4]:
            print("\n  {} (threads={}, {}s)...".format(
                description, threads, BENCHMARK_DURATION))
            r = run_benchmark(SSD_DIR, "test_performance_ssd", model_name,
                            "test_performance_ssd.list", threads, BENCHMARK_DURATION)
            if r["fps"] > 0:
                speedup = r["fps"] / CPU_BASELINE["avg_fps"]
                print("    FPS: {:.2f} | E2E: {:.2f}ms | DPU: {:.2f}ms | Speedup: {:.1f}x".format(
                    r["fps"], r["e2e_mean_ms"], r["dpu_mean_ms"], speedup))
            else:
                print("    FAILED")
            results.append(r)

    # ResNet50 Classification
    if os.path.exists(CLS_DIR):
        for threads in [1, 4]:
            print("\n  ResNet50 Classification (threads={}, {}s)...".format(
                threads, BENCHMARK_DURATION))
            r = run_benchmark(CLS_DIR, "test_performance_classification", "resnet50",
                            "test_performance_classification.list", threads, BENCHMARK_DURATION)
            if r["fps"] > 0:
                speedup = r["fps"] / CPU_BASELINE["avg_fps"]
                print("    FPS: {:.2f} | Speedup: {:.1f}x".format(r["fps"], speedup))
            results.append(r)

    return results


# ============================================================
# Part 2: DPU Detection on Test Images
# ============================================================
def parse_detections(text, fw, fh):
    """Parse test_jpeg_ssd output: RESULT: class_id x1 y1 x2 y2 confidence"""
    dets = []
    for line in text.strip().split('\n'):
        if 'RESULT:' not in line:
            continue
        try:
            p = line.split('RESULT:')[1].strip().split()
            if len(p) >= 6:
                cid = int(p[0])
                x1 = max(0, int(float(p[1])))
                y1 = max(0, int(float(p[2])))
                x2 = min(fw, int(float(p[3])))
                y2 = min(fh, int(float(p[4])))
                conf = float(p[5])
                if conf >= CONF_THRESHOLD:
                    dets.append({
                        "class_id": cid,
                        "confidence": conf,
                        "box": (x1, y1, x2, y2)
                    })
        except:
            continue
    return dets


def draw_detections(frame, dets, model_name, image_name):
    """Draw bounding boxes and info panel on image"""
    out = frame.copy()

    for i, d in enumerate(dets):
        x1, y1, x2, y2 = d["box"]
        c = COLORS[i % len(COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        label = "class_{}: {:.0%}".format(d["class_id"], d["confidence"])
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, y1-lh-10), (x1+lw+5, y1), c, -1)
        cv2.putText(out, label, (x1+2, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.rectangle(out, (0, 0), (450, 90), (0, 0, 0), -1)
    cv2.putText(out, "DPU Inference | {}".format(model_name),
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(out, "Image: {}".format(image_name),
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(out, "Detections: {}".format(len(dets)),
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return out


def run_image_detection():
    """Run DPU detection on all test images using multiple models"""
    print("\n" + "=" * 60)
    print("PART 2: DPU DETECTION ON TEST IMAGES")
    print("All inference runs on FPGA DPU (not CPU)")
    print("=" * 60)

    frames_dir = os.path.join(RESULTS_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    if not os.path.exists(TEST_IMAGES_DIR):
        print("No test images directory: {}".format(TEST_IMAGES_DIR))
        return None

    images = sorted([f for f in os.listdir(TEST_IMAGES_DIR)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    if not images:
        print("No test images found in {}".format(TEST_IMAGES_DIR))
        return None

    print("Found {} test images".format(len(images)))

    models = ["ssd_mobilenet_v2", "ssd_adas_pruned_0_95"]
    all_results = {}

    for model_name in models:
        print("\n--- Model: {} ---".format(model_name))
        model_results = []
        total_dets = 0
        latencies = []

        for img_name in images:
            img_path = os.path.abspath(os.path.join(TEST_IMAGES_DIR, img_name))

            img = cv2.imread(img_path)
            if img is None:
                print("  Cannot read: {}".format(img_name))
                continue
            h, w = img.shape[:2]

            t0 = time.time()
            try:
                result = subprocess.run(
                    [SSD_JPEG_BINARY, model_name, img_path],
                    capture_output=True, text=True, timeout=10)
                all_text = result.stdout + "\n" + result.stderr
                dets = parse_detections(all_text, w, h)
            except:
                dets = []
            elapsed = (time.time() - t0) * 1000
            latencies.append(elapsed)
            total_dets += len(dets)

            # Save annotated image
            annotated = draw_detections(img, dets, model_name, img_name)
            out_name = "{}_{}.jpg".format(model_name, os.path.splitext(img_name)[0])
            cv2.imwrite(os.path.join(frames_dir, out_name), annotated)

            if dets:
                det_str = ", ".join(
                    ["class_{}({:.0%})".format(d["class_id"], d["confidence"])
                     for d in dets[:4]])
                print("  {} | {:.0f}ms | {}".format(img_name, elapsed, det_str))
            else:
                print("  {} | {:.0f}ms | no detections".format(img_name, elapsed))

            model_results.append({
                "image": img_name,
                "latency_ms": round(elapsed, 2),
                "detections": len(dets),
                "details": [{"class_id": d["class_id"],
                             "confidence": round(d["confidence"], 3),
                             "box": list(d["box"])} for d in dets]
            })

        avg_lat = np.mean(latencies) if latencies else 0
        all_results[model_name] = {
            "total_images": len(images),
            "total_detections": total_dets,
            "avg_latency_ms": round(avg_lat, 2),
            "per_image": model_results
        }
        print("  Total: {} detections across {} images".format(total_dets, len(images)))

    return all_results


# ============================================================
# Part 3: Final Report
# ============================================================
def generate_report(benchmarks, image_results):
    """Generate and save complete performance report"""
    print("\n" + "=" * 60)
    print("PART 3: FINAL PERFORMANCE REPORT")
    print("=" * 60)

    report = []
    report.append("=" * 60)
    report.append("HARDWARE-ACCELERATED CNN INFERENCE")
    report.append("PERFORMANCE REPORT")
    report.append("=" * 60)
    report.append("")
    report.append("Date:        {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    report.append("Platform:    Kria KV260 (Zynq UltraScale+ MPSoC)")
    report.append("CPU:         ARM Cortex-A53 @ 1.2 GHz (quad-core)")
    report.append("DPU:         DPUCZDX8G @ 300 MHz")
    report.append("DPU Arch:    DPUCZDX8G_ISA0_B4096_MAX_BG2")
    report.append("Fingerprint: 0x1000020f6014407")
    report.append("Framework:   Vitis-AI 1.4")
    report.append("")

    # CPU Baseline
    report.append("=" * 60)
    report.append("CPU BASELINE (ARM Cortex-A53 + OpenCV DNN)")
    report.append("=" * 60)
    report.append("Model:          {}".format(CPU_BASELINE["model"]))
    report.append("Input:          {}".format(CPU_BASELINE["input"]))
    report.append("Frames:         {}".format(CPU_BASELINE["num_frames"]))
    report.append("FPS:            {:.2f}".format(CPU_BASELINE["avg_fps"]))
    report.append("Latency:        {:.2f} ms".format(CPU_BASELINE["avg_latency"]))
    report.append("  Preprocess:   {:.2f} ms ({:.1f}%)".format(
        CPU_BASELINE["avg_preprocess"],
        CPU_BASELINE["avg_preprocess"] / CPU_BASELINE["avg_latency"] * 100))
    report.append("  Inference:    {:.2f} ms ({:.1f}%)".format(
        CPU_BASELINE["avg_inference"],
        CPU_BASELINE["avg_inference"] / CPU_BASELINE["avg_latency"] * 100))
    report.append("  Postprocess:  {:.2f} ms ({:.1f}%)".format(
        CPU_BASELINE["avg_postprocess"],
        CPU_BASELINE["avg_postprocess"] / CPU_BASELINE["avg_latency"] * 100))
    report.append("Objects:        person({}), chair({})".format(
        CPU_BASELINE["objects"]["person"], CPU_BASELINE["objects"]["chair"]))
    report.append("")

    # DPU Benchmarks
    report.append("=" * 60)
    report.append("DPU ACCELERATED (FPGA + Vitis-AI)")
    report.append("=" * 60)
    report.append("")
    report.append("{:<30s} {:>6s} {:>8s} {:>10s} {:>10s} {:>10s}".format(
        "Model", "Thrd", "FPS", "E2E(ms)", "DPU(ms)", "Speedup"))
    report.append("-" * 75)

    for r in benchmarks:
        if r.get("fps", 0) > 0:
            speedup = r["fps"] / CPU_BASELINE["avg_fps"]
            report.append("{:<30s} {:>6d} {:>8.2f} {:>10.2f} {:>10.2f} {:>9.1f}x".format(
                r["model"], r.get("threads", 1), r["fps"],
                r.get("e2e_mean_ms", 0), r.get("dpu_mean_ms", 0), speedup))

    report.append("")

    # Key comparison
    dpu_ssd = None
    for r in benchmarks:
        if r["model"] == "ssd_mobilenet_v2" and r.get("threads") == 1 and r["fps"] > 0:
            dpu_ssd = r
            break

    report.append("=" * 60)
    report.append("KEY COMPARISON (SSD MobileNet V2, single-thread)")
    report.append("=" * 60)
    report.append("")

    if dpu_ssd:
        dpu_fps = dpu_ssd["fps"]
        dpu_e2e = dpu_ssd.get("e2e_mean_ms", 1000.0 / dpu_fps)
        dpu_inf = dpu_ssd.get("dpu_mean_ms", 0)
        sp_fps = dpu_fps / CPU_BASELINE["avg_fps"]
        sp_lat = CPU_BASELINE["avg_latency"] / dpu_e2e if dpu_e2e > 0 else 0
        sp_inf = CPU_BASELINE["avg_inference"] / dpu_inf if dpu_inf > 0 else 0

        report.append("  {:<25s} {:>12s} {:>12s} {:>10s}".format(
            "Metric", "CPU", "DPU", "Speedup"))
        report.append("  " + "-" * 60)
        report.append("  {:<25s} {:>12.2f} {:>12.2f} {:>9.1f}x".format(
            "FPS", CPU_BASELINE["avg_fps"], dpu_fps, sp_fps))
        report.append("  {:<25s} {:>12.2f} {:>12.2f} {:>9.1f}x".format(
            "E2E Latency (ms)", CPU_BASELINE["avg_latency"], dpu_e2e, sp_lat))
        report.append("  {:<25s} {:>12.2f} {:>12.2f} {:>9.1f}x".format(
            "Pure Inference (ms)", CPU_BASELINE["avg_inference"], dpu_inf, sp_inf))

    report.append("")

    # Image results
    if image_results:
        report.append("=" * 60)
        report.append("DPU IMAGE DETECTION RESULTS")
        report.append("=" * 60)
        for model_name, data in image_results.items():
            report.append("")
            report.append("Model: {}".format(model_name))
            report.append("  Images:     {}".format(data["total_images"]))
            report.append("  Detections: {}".format(data["total_detections"]))
            for img_data in data["per_image"]:
                if img_data["detections"] > 0:
                    dets_str = ", ".join(
                        ["class_{}({:.0%})".format(d["class_id"], d["confidence"])
                         for d in img_data["details"][:4]])
                    report.append("    {}: {}".format(img_data["image"], dets_str))

    report.append("")

    # Conclusion
    successful_1t = [r for r in benchmarks if r.get("fps", 0) > 0 and r.get("threads") == 1]
    all_successful = [r for r in benchmarks if r.get("fps", 0) > 0]

    if successful_1t:
        min_sp = min([r["fps"] / CPU_BASELINE["avg_fps"] for r in successful_1t])
        max_sp = max([r["fps"] / CPU_BASELINE["avg_fps"] for r in successful_1t])
    else:
        min_sp = max_sp = 0

    best_fps = max([r["fps"] for r in all_successful], default=0)

    report.append("=" * 60)
    report.append("CONCLUSION")
    report.append("=" * 60)
    report.append("")
    report.append("  Project Requirement:     2x speedup minimum")
    report.append("  Achieved (1-thread):     {:.1f}x - {:.1f}x".format(min_sp, max_sp))
    report.append("  Best FPS:                {:.2f} (multi-threaded)".format(best_fps))
    report.append("  Status:                  {}".format(
        "SUCCESS!" if min_sp >= 2.0 else "NEEDS REVIEW"))
    report.append("")
    report.append("  HW/SW Co-design:")
    report.append("    - ARM CPU: preprocessing, postprocessing, control")
    report.append("    - FPGA DPU: CNN inference (INT8 quantized xmodel)")
    report.append("")
    report.append("  FPGA DPU enables real-time inference ({:.0f} FPS)".format(best_fps))
    report.append("  vs CPU-only ({:.2f} FPS), achieving {:.1f}x-{:.1f}x".format(
        CPU_BASELINE["avg_fps"], min_sp, max_sp))
    report.append("  speedup on embedded edge hardware.")
    report.append("")
    report.append("=" * 60)

    report_text = "\n".join(report)
    print(report_text)

    # Save files
    with open(os.path.join(RESULTS_DIR, "dpu_report.txt"), "w") as f:
        f.write(report_text)

    json_data = {
        "metadata": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "platform": "Kria KV260 (Zynq UltraScale+ MPSoC)",
            "cpu": "ARM Cortex-A53 @ 1.2 GHz",
            "dpu": "DPUCZDX8G @ 300 MHz",
            "dpu_arch": "DPUCZDX8G_ISA0_B4096_MAX_BG2",
            "fingerprint": "0x1000020f6014407",
            "framework": "Vitis-AI 1.4"
        },
        "cpu_baseline": CPU_BASELINE,
        "dpu_benchmarks": benchmarks,
        "image_detection": image_results
    }
    with open(os.path.join(RESULTS_DIR, "dpu_results.json"), "w") as f:
        json.dump(json_data, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "benchmark_results.csv"), "w") as f:
        f.write("model,threads,fps,e2e_ms,dpu_ms,speedup\n")
        for r in benchmarks:
            if r.get("fps", 0) > 0:
                sp = r["fps"] / CPU_BASELINE["avg_fps"]
                f.write("{},{},{:.2f},{:.2f},{:.2f},{:.1f}\n".format(
                    r["model"], r.get("threads", 1), r["fps"],
                    r.get("e2e_mean_ms", 0), r.get("dpu_mean_ms", 0), sp))

    print("\nResults saved to: {}/".format(RESULTS_DIR))
    print("  dpu_report.txt        - Performance report")
    print("  dpu_results.json      - Complete JSON data")
    print("  benchmark_results.csv - Benchmark CSV")
    print("  frames/               - Annotated detection images")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("DPU ACCELERATED OBJECT DETECTION")
    print("Kria KV260 | DPUCZDX8G @ 300MHz | Vitis-AI 1.4")
    print("Date: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    benchmarks = run_all_benchmarks()
    image_results = run_image_detection()
    generate_report(benchmarks, image_results)


if __name__ == "__main__":
    main()
