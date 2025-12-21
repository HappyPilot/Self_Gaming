# ONNX Export (YOLO)

Use `tools/export_onnx.py` to convert Ultralytics YOLO weights to ONNX.

## Prereqs
- `ultralytics` installed in the environment (or run inside the container that has it).
- Input weights available locally (for example, `/mnt/ssd/models/yolo/yolo11n.pt`).

## Export
```bash
python3 tools/export_onnx.py /mnt/ssd/models/yolo/yolo11n.pt \
  --output /mnt/ssd/models/yolo \
  --imgsz 416 \
  --device cuda:0
```

This writes the ONNX file into `/mnt/ssd/models/yolo/` with the default export name.
If you want a specific filename:
```bash
python3 tools/export_onnx.py /mnt/ssd/models/yolo/yolo11n.pt \
  --output /mnt/ssd/models/yolo/yolo11n_416_fp32.onnx \
  --imgsz 416 \
  --device cuda:0
```

## Verify
```bash
python -c "import onnx; onnx.load('/mnt/ssd/models/yolo/yolo11n_416_fp32.onnx')"
```

## Notes
- `OBJECT_MODEL_PATH` in `config/defaults.env` expects an ONNX model path when `OBJECT_DETECTOR_BACKEND=onnx`.
- `perception_agent` uses `DETECTOR_BACKEND` / `YOLO11_*` separately; this doc targets the object detection agent's ONNX path.
- Use `--dynamic` only if you need variable input sizes; static inputs are generally faster.
- If your runtime needs a newer opset (for example TensorRT), try `--opset 17`.
