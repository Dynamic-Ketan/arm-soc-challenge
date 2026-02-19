#!/usr/bin/env python3
import cv2
import time
import numpy as np
import os
import json
from collections import Counter
from datetime import datetime

# Configuration
NUM_FRAMES = 100
CAMERA_INDEX = 0
CONF_THRESHOLD = 0.5
DEBUG_MODE = True
SAVE_FRAMES = True
SAVE_RESULTS = True
OUTPUT_DIR = "cpu_results"

# COCO class labels
COCO_LABELS = {
    0: "background", 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle",
    5: "airplane", 6: "bus", 7: "train", 8: "truck", 9: "boat",
    10: "traffic light", 11: "fire hydrant", 13: "stop sign", 14: "parking meter",
    15: "bench", 16: "bird", 17: "cat", 18: "dog", 19: "horse",
    20: "sheep", 21: "cow", 22: "elephant", 23: "bear", 24: "zebra",
    25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag",
    32: "tie", 33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard",
    37: "sports ball", 38: "kite", 39: "baseball bat", 40: "baseball glove",
    41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle",
    46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon",
    51: "bowl", 52: "banana", 53: "apple", 54: "sandwich", 55: "orange",
    56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut",
    61: "cake", 62: "chair", 63: "couch", 64: "potted plant", 65: "bed",
    67: "dining table", 70: "toilet", 72: "tv", 73: "laptop", 74: "mouse",
    75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave",
    79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book",
    85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear",
    89: "hair drier", 90: "toothbrush"
}

COLORS = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 255, 0), (255, 128, 0)
]


class CPUObjectDetector:
    """CPU-based object detection using OpenCV DNN"""

    def __init__(self, model_path, config_path):
        print("=" * 60)
        print("CPU BASELINE IMPLEMENTATION")
        print("Model: SSD MobileNet V2 (COCO)")
        print("Backend: OpenCV DNN on ARM Cortex-A53")
        print("=" * 60)

        assert os.path.exists(model_path), "Model not found: {}".format(model_path)
        assert os.path.exists(config_path), "Config not found: {}".format(config_path)

        print("Loading model: {}".format(model_path))
        print("Loading config: {}".format(config_path))

        self.net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.latencies = []
        self.preprocessing_times = []
        self.inference_times = []
        self.postprocessing_times = []
        self.detection_counts = []
        self.all_detections = []
        self.frame_results = []

        print("Model loaded successfully (CPU)")

    def preprocess(self, frame):
        t0 = time.time()
        blob = cv2.dnn.blobFromImage(
            frame, size=(300, 300),
            mean=(104, 177, 123),
            scalefactor=1.0,
            swapRB=False, crop=False
        )
        self.preprocessing_times.append((time.time() - t0) * 1000)
        return blob

    def inference(self, blob):
        self.net.setInput(blob)
        t0 = time.time()
        detections = self.net.forward()
        self.inference_times.append((time.time() - t0) * 1000)
        return detections

    def postprocess(self, detections, frame_shape, threshold):
        t0 = time.time()
        h, w = frame_shape[:2]
        results = []

        if DEBUG_MODE and len(self.latencies) == 0:
            print("\nDEBUG: detections shape = {}".format(detections.shape))

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > threshold:
                class_id = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                class_name = COCO_LABELS.get(class_id, "unknown_{}".format(class_id))

                results.append({
                    "confidence": float(confidence),
                    "class_id": class_id,
                    "class_name": class_name,
                    "box": box.astype(int).tolist()
                })
                self.all_detections.append(class_name)

        self.postprocessing_times.append((time.time() - t0) * 1000)
        return results

    def detect(self, frame):
        t_total = time.time()
        blob = self.preprocess(frame)
        detections = self.inference(blob)
        results = self.postprocess(detections, frame.shape, CONF_THRESHOLD)
        total_latency = (time.time() - t_total) * 1000
        self.latencies.append(total_latency)
        self.detection_counts.append(len(results))
        return results, total_latency

    def draw_detections(self, frame, results, fps, frame_num):
        """Draw bounding boxes on frame"""
        output = frame.copy()

        for i, det in enumerate(results):
            x1, y1, x2, y2 = det["box"]
            color = COLORS[i % len(COLORS)]

            # Bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Label with background
            label = "{}: {:.0%}".format(det["class_name"], det["confidence"])
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(output, (x1, y1-lh-10), (x1+lw+5, y1), color, -1)
            cv2.putText(output, label, (x1+2, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Info panel
        cv2.rectangle(output, (0, 0), (350, 100), (0, 0, 0), -1)
        cv2.putText(output, "CPU Mode | FPS: {:.1f}".format(fps),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(output, "Objects: {}".format(len(results)),
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(output, "Frame: {}/{}".format(frame_num, NUM_FRAMES),
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return output

    def get_statistics(self):
        return {
            "avg_latency": float(np.mean(self.latencies)),
            "std_latency": float(np.std(self.latencies)),
            "min_latency": float(np.min(self.latencies)),
            "max_latency": float(np.max(self.latencies)),
            "avg_fps": float(1000 / np.mean(self.latencies)),
            "avg_preprocess": float(np.mean(self.preprocessing_times)),
            "avg_inference": float(np.mean(self.inference_times)),
            "avg_postprocess": float(np.mean(self.postprocessing_times)),
            "total_frames": len(self.latencies),
            "total_detections": sum(self.detection_counts),
            "avg_detections": float(np.mean(self.detection_counts)),
            "max_detections": max(self.detection_counts) if self.detection_counts else 0
        }

    def get_detection_summary(self):
        return Counter(self.all_detections)


def benchmark_cpu():
    model_path = "ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"
    config_path = "ssd_mobilenet_v2_coco_2018_03_29.pbtxt"

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    frames_dir = os.path.join(OUTPUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    detector = CPUObjectDetector(model_path, config_path)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Camera not available, using synthetic frames")
        use_camera = False
        cam_info = "synthetic 480x640"
    else:
        use_camera = True
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cam_info = "{}x{}".format(width, height)
        print("Camera resolution: {}".format(cam_info))

    print("\nRunning CPU benchmark ({} frames)".format(NUM_FRAMES))
    print("-" * 60)

    frame_count = 0
    start_time = time.time()

    while frame_count < NUM_FRAMES:
        if use_camera:
            ret, frame = cap.read()
            if not ret:
                continue
        else:
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        results, latency = detector.detect(frame)
        fps = 1000 / latency if latency > 0 else 0
        frame_count += 1

        # Save annotated frame
        if SAVE_FRAMES and (frame_count <= 10 or frame_count % 10 == 0 or results):
            annotated = detector.draw_detections(frame, results, fps, frame_count)
            annotated = cv2.resize(annotated, (640, 480))
            cv2.imwrite("{}/frame_{:03d}.jpg".format(frames_dir, frame_count), annotated)

        # Store per-frame results
        detector.frame_results.append({
            "frame": frame_count,
            "latency_ms": round(latency, 2),
            "fps": round(fps, 2),
            "detections": len(results),
            "objects": [r["class_name"] for r in results]
        })

        # Print progress
        if results:
            objects = ["{}({:.0%})".format(r["class_name"], r["confidence"]) for r in results]
            objects_str = ", ".join(objects[:4])
            if len(results) > 4:
                objects_str += " +{} more".format(len(results) - 4)
        else:
            objects_str = "none"

        print("Frame {:03d} | Latency: {:6.1f}ms | FPS: {:4.1f} | Objects: {}".format(
            frame_count, latency, fps, objects_str))

    total_time = time.time() - start_time

    if use_camera:
        cap.release()

    stats = detector.get_statistics()
    detection_summary = detector.get_detection_summary()

    # Print report
    report = []
    report.append("")
    report.append("=" * 60)
    report.append("CPU BASELINE PERFORMANCE REPORT")
    report.append("=" * 60)
    report.append("Date:                  {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    report.append("Model:                 SSD MobileNet V2 (COCO)")
    report.append("Platform:              ARM Cortex-A53 (CPU only)")
    report.append("Camera:                {}".format(cam_info))
    report.append("Total Frames:          {}".format(stats["total_frames"]))
    report.append("Total Time:            {:.1f} seconds".format(total_time))
    report.append("Average Latency:       {:.2f} +/- {:.2f} ms".format(
        stats["avg_latency"], stats["std_latency"]))
    report.append("Min Latency:           {:.2f} ms".format(stats["min_latency"]))
    report.append("Max Latency:           {:.2f} ms".format(stats["max_latency"]))
    report.append("Average FPS:           {:.2f}".format(stats["avg_fps"]))
    report.append("-" * 60)
    report.append("Latency Breakdown:")
    report.append("  Preprocessing:       {:.2f} ms ({:.1f}%)".format(
        stats["avg_preprocess"],
        stats["avg_preprocess"] / stats["avg_latency"] * 100))
    report.append("  Inference (CPU):     {:.2f} ms ({:.1f}%)".format(
        stats["avg_inference"],
        stats["avg_inference"] / stats["avg_latency"] * 100))
    report.append("  Postprocessing:      {:.2f} ms ({:.1f}%)".format(
        stats["avg_postprocess"],
        stats["avg_postprocess"] / stats["avg_latency"] * 100))
    report.append("-" * 60)
    report.append("Total Detections:      {}".format(stats["total_detections"]))
    report.append("Avg Detections/Frame:  {:.2f}".format(stats["avg_detections"]))
    report.append("-" * 60)
    if detection_summary:
        report.append("Objects Detected:")
        for obj, count in detection_summary.most_common(10):
            bar = "#" * min(count, 30)
            report.append("  {:20s}: {:4d}  {}".format(obj, count, bar))
    else:
        report.append("No objects detected")
    report.append("=" * 60)

    report_text = "\n".join(report)
    print(report_text)

    # Save results
    if SAVE_RESULTS:
        # Save text report
        with open("{}/cpu_report.txt".format(OUTPUT_DIR), "w") as f:
            f.write("CPU BASELINE PERFORMANCE REPORT\n")
            f.write(report_text)

        # Save JSON results
        json_data = {
            "metadata": {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": "SSD MobileNet V2 (COCO)",
                "platform": "ARM Cortex-A53 (CPU only)",
                "board": "Kria KV260",
                "camera": cam_info,
                "num_frames": NUM_FRAMES,
                "conf_threshold": CONF_THRESHOLD
            },
            "performance": stats,
            "detection_summary": dict(detection_summary),
            "per_frame": detector.frame_results
        }

        with open("{}/cpu_results.json".format(OUTPUT_DIR), "w") as f:
            json.dump(json_data, f, indent=2)

        # Save CSV
        with open("{}/cpu_latency.csv".format(OUTPUT_DIR), "w") as f:
            f.write("frame,latency_ms,fps,detections,objects\n")
            for r in detector.frame_results:
                f.write("{},{},{},{},{}\n".format(
                    r["frame"], r["latency_ms"], r["fps"],
                    r["detections"], "|".join(r["objects"])))

        print("\nResults saved to: {}/".format(OUTPUT_DIR))
        print("  cpu_report.txt   - Text report")
        print("  cpu_results.json - JSON data")
        print("  cpu_latency.csv  - Per-frame CSV")
        print("  frames/          - Annotated frames")


if __name__ == "__main__":
    benchmark_cpu()
