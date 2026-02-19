# Model Conversion: TensorFlow → DPU xmodel

## Step-by-Step Guide to compile SSD MobileNet V1 for Kria KV260 DPU

---

## Step 1: Clone Vitis-AI Repository

```bash
# On your host PC (with Docker installed)
git clone --recurse-submodules https://github.com/Xilinx/Vitis-AI.git
cd Vitis-AI
```

## Step 2: Start Vitis-AI Docker Container
```
./docker_run.sh xilinx/vitis-ai-gpu:3.5
You will see:

text

==========================================
__      ___ _   _                _    ___
\ \    / (_) | (_)              / \  |_ _|
 \ \  / / _| |_ _ ___         / _ \  | |
  \ \/ / | | __| / __|  ___  / ___ \ | |
   \  /  | | |_| \__ \ |___|| /   \ \|___|
    \/   |_|\__|_|___/       |_|   |_|

==========================================
Vitis-AI /workspace >
```

## Step 3: Activate TensorFlow Environment
```
conda activate vitis-ai-tensorflow
```

(vitis-ai-tensorflow) Vitis-AI /workspace >

## Snap

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/c84d0603-f04b-4361-9078-ae08a0d3b73e" />


## Step 4: Create Project Directory

```
mkdir -p /workspace/fpga_arm/ssd_mobilenet_v1_project
cd /workspace/fpga_arm/ssd_mobilenet_v1_project
```

## Step 5: Download Pre-quantized Model from Vitis-AI Model Zoo

```
# Download SSD MobileNet V1 (pre-quantized by Xilinx)
wget https://www.xilinx.com/bin/public/openDownload?filename=tf_ssdmobilenetv1_coco-zcu102_zcu104_kv260-r3.5.0.tar.gz -O tf_ssdmobilenetv1_3.5.tar.gz

# Extract
tar -xzf tf_ssdmobilenetv1_3.5.tar.gz
ls tf_ssdmobilenetv1_3.5/quantized/
```

Expected output:

```
quantize_eval_model.pb
```

This is the pre-quantized INT8 model ready for compilation.

## Step 6: Set Environment Variables

```
export BUILD=/workspace/fpga_arm/ssd_mobilenet_v1_project/build
export LOG=/workspace/fpga_arm/ssd_mobilenet_v1_project/logs
mkdir -p ${BUILD}/compiled_model
mkdir -p ${LOG}
```

## Step 7: Verify Architecture File Exists

```
cat /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json
```

Expected output:

```

{
    "target": "DPUCZDX8G",
}
```

This tells the compiler to generate instructions for the KV260 DPU.

## Step 8: Compile the Model

```
vai_c_tensorflow \
    --frozen_pb tf_ssdmobilenetv1_3.5/quantized/quantize_eval_model.pb \
    --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
    --output_dir ${BUILD}/compiled_model \
    --net_name ssd_mobilenet_v1_kv260 \
    2>&1 | tee ${LOG}/compile_kv260.log
```

Compilation Output:

```
[INFO] Namespace(batchsize=1, inputs_shape=None, layout='NHWC', ...)
[INFO] tensorflow model: quantize_eval_model.pb
[INFO] parse raw model     :100%|██████████| 161/161
[INFO] infer shape (NHWC)  :100%|██████████| 269/269
[INFO] perform level-0 opt :100%|██████████| 3/3
[INFO] perform level-1 opt :100%|██████████| 7/7
[INFO] generate xmodel     :100%|██████████| 173/173
[UNILOG][INFO] Compile mode: dpu
[UNILOG][INFO] Target architecture: DPUCZDX8G_ISA1_B4096
[UNILOG][INFO] Graph name: quantize_eval_model, with op num: 361
[UNILOG][INFO] Total device subgraph number 4, DPU subgraph number 1
[UNILOG][INFO] Compile done.
```

*The quantized SSD MobileNet-V1 model was compiled using the Vitis AI compiler targeting the KV260 DPU (DPUCZDX8G architecture).*

## Snap

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/b2ae0de0-e9b8-4e71-87d5-34c2a3a785fd" />


