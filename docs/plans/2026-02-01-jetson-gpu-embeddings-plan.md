# Jetson GPU Embeddings Reliability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restore GPU-backed embeddings on Jetson while keeping the current embeddings-first architecture stable.

**Architecture:** First validate the Jetson GPU stack (JetPack/CUDA/PyTorch) inside the running container with repeatable diagnostics. If PyTorch GPU is unhealthy, either align the container base to the correct L4T/PyTorch build (Plan A) or fall back to a TensorRT encoder with adjusted embedding dimension and re-collected centroids (Plan B).

**Tech Stack:** Docker (Jetson), PyTorch, CUDA/cuDNN, TensorRT, MQTT agents.

---

### Task 1: Add repeatable GPU diagnostics scripts

**Files:**
- Create: `tools/jetson/gpu_diag.sh`
- Create: `tools/jetson/check_torch_cuda.py`

**Step 1: Write the failing test**
- Not applicable (ops script). Use runtime checks below.

**Step 2: Create `tools/jetson/gpu_diag.sh`**
```bash
#!/usr/bin/env bash
set -euo pipefail

printf "HOST: %s\n" "$(hostname)"
printf "DATE: %s\n" "$(date)"

if [ -f /etc/nv_tegra_release ]; then
  echo "--- /etc/nv_tegra_release ---"
  cat /etc/nv_tegra_release
fi

echo "--- CUDA version ---"
cat /usr/local/cuda/version.txt 2>/dev/null || true

echo "--- JetPack packages ---"
dpkg -l | grep -E "nvidia-jetpack|cuda|cudnn|tensorrt" || true

echo "--- tegrastats (5s) ---"
if command -v tegrastats >/dev/null 2>&1; then
  timeout 5s sudo -n tegrastats --interval 1000 || true
else
  echo "tegrastats not found"
fi
```

**Step 3: Create `tools/jetson/check_torch_cuda.py`**
```python
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
print("cudnn", torch.backends.cudnn.version())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    x = torch.randn(128, 128, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("cuda_matmul_ok", y.shape)
```

**Step 4: Run scripts**
Run:
```bash
bash tools/jetson/gpu_diag.sh
python3 tools/jetson/check_torch_cuda.py
```
Expected: CUDA available = True and matmul succeeds.

**Step 5: Commit**
```bash
git add tools/jetson/gpu_diag.sh tools/jetson/check_torch_cuda.py
git commit -m "Add Jetson GPU diagnostics"
```

---

### Task 2: Validate CUDA inside the running vl_jepa container

**Files:**
- No code changes

**Step 1: Run container check**
```bash
sudo -n docker exec -it vl_jepa_agent python3 /app/tools/jetson/check_torch_cuda.py
```
Expected: `cuda_available True`, `cuda_matmul_ok` succeeds.

**Step 2: Record failure mode**
If it fails (CUBLAS/alloc), capture:
```bash
sudo -n docker logs --tail 200 vl_jepa_agent
```

**Step 3: Decision**
- If CUDA works: proceed to Task 3 (GPU embeddings test)
- If CUDA fails: proceed to Task 4 (Plan A: align container stack)

---

### Task 3: GPU embeddings smoke test (Plan A success path)

**Files:**
- Modify: `config/jetson.env`

**Step 1: Set GPU for SigLIP2**
```env
VL_JEPA_DEVICE=cuda
VL_JEPA_FP16=1
```

**Step 2: Restart embeddings services**
```bash
sudo -n docker compose up -d --force-recreate vl_jepa embedding_guard scene policy
```

**Step 3: Verify embeddings flow**
```bash
timeout 5s mosquitto_sub -h 127.0.0.1 -t vision/embeddings -C 1
```
Expected: JSON payload with embedding array.

**Step 4: Verify GPU load**
```bash
timeout 5s sudo -n tegrastats --interval 1000
```
Expected: `GR3D_FREQ` > 0% during embedding generation.

**Step 5: Commit**
```bash
git add config/jetson.env
git commit -m "Enable GPU embeddings"
```

---

### Task 4: Align container to JetPack/PyTorch (Plan A failure path)

**Files:**
- Modify: `docker/vl_jepa/Dockerfile`
- Modify: `docker/vl_jepa/requirements.txt` (if needed)

**Step 1: Determine JetPack/L4T version**
Use output from Task 1 to choose matching base image.

**Step 2: Switch base image to NVIDIA L4T PyTorch**
Example:
```dockerfile
FROM nvcr.io/nvidia/l4t-pytorch:rXX.YY-pth2.Z
```
(Use the tag that matches your JetPack/L4T version.)

**Step 3: Rebuild and restart**
```bash
sudo -n docker compose build vl_jepa
sudo -n docker compose up -d --force-recreate vl_jepa embedding_guard scene policy
```

**Step 4: Re-run Task 2 and Task 3**
Proceed only if CUDA matmul succeeds.

**Step 5: Commit**
```bash
git add docker/vl_jepa/Dockerfile docker/vl_jepa/requirements.txt
git commit -m "Align vl_jepa container to JetPack CUDA"
```

---

### Task 5: Fallback Plan B — TensorRT embeddings (if Plan A fails)

**Files:**
- Modify: `config/jetson.env`
- Modify: `agents/embedding_guard_agent.py` (if dimension changes)

**Step 1: Switch to TensorRT backend**
```env
VL_JEPA_BACKEND=tensorrt
VL_JEPA_ENGINE_PATH=/mnt/ssd/models/vl_jepa/dinov2_small_224_f1_fp16.engine
VL_JEPA_INPUT_SIZE=224
```

**Step 2: Confirm embedding dim**
Run a one-off script in container to print embedding size.
If engine outputs 384-dim, set:
```env
EMBEDDING_GUARD_DIM=384
```

**Step 3: Re-collect centroids**
```bash
EMBEDDING_GUARD_MODE=collect_game
EMBEDDING_GUARD_MODE=collect_non
```
(Collect ~40 samples each, then return to `run`.)

**Step 4: Restart agents**
```bash
sudo -n docker compose up -d --force-recreate vl_jepa embedding_guard scene policy
```

**Step 5: Verify flow**
```bash
timeout 5s mosquitto_sub -h 127.0.0.1 -t vision/embeddings -C 1
```
Expected: embeddings array present.

**Step 6: Commit**
```bash
git add config/jetson.env agents/embedding_guard_agent.py
git commit -m "Switch embeddings to TensorRT"
```

---

### Task 6: Success validation

**Files:**
- None

**Step 1: Verify understanding/controls**
```bash
timeout 5s mosquitto_sub -h 127.0.0.1 -t progress/understanding -C 1
```
Expected: embeddings block present, stable `locations` updates.

**Step 2: Verify actuation**
```bash
timeout 5s mosquitto_sub -h 127.0.0.1 -t act/result -C 1
```

**Step 3: Record tegrastats**
```bash
timeout 5s sudo -n tegrastats --interval 1000
```
Expected: `GR3D_FREQ` > 0% if GPU path is active.

**Step 4: Commit notes**
Update `docs/plans/2026-02-01-embeddings-first-vision-design.md` with the chosen path and results.

