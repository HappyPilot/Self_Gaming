import os
import sys
import glob
import cv2
import numpy as np
import shutil
import random
import json
import base64
import textwrap
from pathlib import Path

# Add local repo to path for Git Clone installation method (image + host)
ULTRA_PATHS = ["/app/ultralytics_repo", "/opt/ultralytics_repo"]
for ultra_path in ULTRA_PATHS:
    if os.path.isdir(ultra_path) and ultra_path not in sys.path:
        sys.path.append(ultra_path)

from ultralytics import YOLO

# --- CONFIG ---
DATASET_DIR = "/mnt/ssd/datasets/episodes"
OUTPUT_DIR = "/mnt/ssd/datasets/yolo_auto"
MODEL_PATH = "/mnt/ssd/models/yolo/yolo11n.pt"
NEW_MODEL_PATH = "/mnt/ssd/models/yolo/yolo11n_poe.pt"
MIN_AREA = 500
MAX_SAMPLES = 200
EPOCHS = 10

def decode_image(b64_str):
    try:
        if not b64_str: return None
        nparr = np.frombuffer(base64.b64decode(b64_str), np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except: return None

def extract_and_label():
    print("🔍 Scanning for episodes (*.json)...")
    files = sorted(glob.glob(f"{DATASET_DIR}/*.json"))
    if not files:
        print("❌ No data found.")
        return False

    print(f"Found {len(files)} files. Processing sample...")
    
    # Prepare directories
    img_dir = Path(OUTPUT_DIR) / "images" / "train"
    lbl_dir = Path(OUTPUT_DIR) / "labels" / "train"
    if Path(OUTPUT_DIR).exists(): shutil.rmtree(OUTPUT_DIR)
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    count = 0
    random.shuffle(files)
    
    # Limit processing to avoid OOM
    for json_file in files[:500]: 
        if count >= MAX_SAMPLES: break
        try:
            with open(json_file, "r") as f:
                record = json.load(f)
            
            # Extract image
            scene = record.get("scene", {})
            # Check possible keys for image
            b64 = scene.get("image_b64") or scene.get("snapshot")
            
            frame = decode_image(b64)
            if frame is None: continue

            # Motion Detection
            fg_mask = back_sub.apply(frame)
            _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            labels = []
            h, w = frame.shape[:2]
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > MIN_AREA:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    cx, cy = x + bw/2, y + bh/2
                    
                    # Filter center (player)
                    if abs(cx - w/2) < w*0.1 and abs(cy - h/2) < h*0.1: continue 
                    
                    nx, ny, nw, nh = cx/w, cy/h, bw/w, bh/h
                    labels.append(f"0 {nx:.4f} {ny:.4f} {nw:.4f} {nh:.4f}")
            
            if labels:
                img_name = f"{Path(json_file).stem}.jpg"
                cv2.imwrite(str(img_dir / img_name), frame)
                with open(lbl_dir / img_name.replace(".jpg", ".txt"), "w") as f:
                    f.write("\n".join(labels))
                count += 1
                if count % 10 == 0: print(f"Generated {count} samples...")
                
        except Exception as e:
            pass # Skip bad files
            
    print(f"✅ Generated {count} labeled samples.")
    return count > 10

def create_yaml():
    yaml_content = textwrap.dedent(f"""
    path: {OUTPUT_DIR}
    train: images/train
    val: images/train
    nc: 1
    names: ['game_entity']
    """)
    with open(f"{OUTPUT_DIR}/dataset.yaml", "w") as f:
        f.write(yaml_content)

def train():
    print("🚀 Starting YOLO training...")
    # Load model (use CPU for training if GPU is busy or unavailable in this container context, but usually CUDA is best)
    # We rely on Ultralytics to auto-select device
    model = YOLO(MODEL_PATH) 
    
    model.train(
        data=f"{OUTPUT_DIR}/dataset.yaml",
        epochs=EPOCHS,
        imgsz=640,
        batch=8,
        project="/mnt/ssd/models/yolo/runs",
        name="poe_auto",
        amp=False # Disable AMP to fix Jetson double free crash
    )
    
    # Export / Save
    print(f"💾 Saving model to {NEW_MODEL_PATH}")
    # Typically best.pt is in runs/poe_auto/weights/best.pt
    # Find the best.pt
    best_pt = Path("/mnt/ssd/models/yolo/runs/poe_auto/weights/best.pt")
    if best_pt.exists():
        shutil.copy(str(best_pt), NEW_MODEL_PATH)
        print("✅ Model updated!")
    else:
        print("⚠️ Could not find best.pt")

if __name__ == "__main__":
    if extract_and_label():
        create_yaml()
        train()
    else:
        print("Not enough data to train.")
