#!/bin/bash
set -euo pipefail

# Optional cache to avoid rebuilds across container recreations
PYDS_CACHE="${PYDS_CACHE:-/mnt/ssd/models/pyds/pyds.so}"

ensure_pyds() {
  if python3 -c "import pyds" >/dev/null 2>&1; then
    echo "[entrypoint] pyds already present"
    return
  fi

  site_dir="$(python3 - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

  if [ -f "$PYDS_CACHE" ]; then
    echo "[entrypoint] restoring pyds from cache $PYDS_CACHE"
    mkdir -p "$site_dir"
    cp "$PYDS_CACHE" "$site_dir"/pyds.so
    if python3 -c "import pyds" >/dev/null 2>&1; then
      echo "[entrypoint] pyds restored from cache"
      return
    fi
    echo "[entrypoint] cache restore failed, will rebuild"
  fi

  echo "[entrypoint] pyds missing, building inside container (requires GPU libs mounted)..."
  workdir="$(mktemp -d /tmp/pyds.XXXX)"
  trap 'rm -rf "${workdir}"' EXIT
  cd "${workdir}"

  git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps
  cd deepstream_python_apps
  git checkout v1.2.2
  git submodule update --init bindings/3rdparty/pybind11 bindings/3rdparty/git-partial-submodule

  sed -i 's/set(PYTHON_MINOR_VERSION 12)/set(PYTHON_MINOR_VERSION 10)/' bindings/CMakeLists.txt
  sed -i 's/set(PYTHON_MINVERS_ALLOWED 12)/set(PYTHON_MINVERS_ALLOWED 10 12)/' bindings/CMakeLists.txt
  sed -i '/NVDS_FRAME_META_NVDSANALYTICS/d' bindings/src/bindnvdsmeta.cpp
  sed -i '/NVDS_OBJ_META_NVDSANALYTICS/d' bindings/src/bindnvdsmeta.cpp

  cd bindings
  mkdir -p build && cd build
  cmake .. -DPYTHON_MAJOR_VERSION=3 -DPYTHON_MINOR_VERSION=10 -DDS_PATH=/opt/nvidia/deepstream/deepstream-7.1
  make -j"$(nproc)"

  cp pyds.so "${site_dir}/"
  mkdir -p "$(dirname "$PYDS_CACHE")" || true
  cp pyds.so "$PYDS_CACHE" || true
  ldconfig
  echo "[entrypoint] pyds build finished"
}

ensure_pyds
exec python3 /app/deepstream_mqtt.py
