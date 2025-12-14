# syntax=docker/dockerfile:1.4
# DeepStream-based perception image (JetPack 6.1 / DeepStream 7.1)
FROM nvcr.io/nvidia/deepstream-l4t:7.1-triton-multiarch

ENV DS_PATH=/opt/nvidia/deepstream/deepstream-7.1

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        python3-pip python3-gi python3-dev \
        git cmake build-essential libglib2.0-dev pkg-config; \
    pip3 install --no-cache-dir paho-mqtt cython

# Build and install pyds at image build time (avoid runtime compilation).
# Requires buildx/BuildKit to bind host DeepStream/tegra libs during build on Jetson.
RUN --mount=type=bind,source=/opt/nvidia/deepstream,target=/opt/nvidia/deepstream,readonly \
    --mount=type=bind,source=/usr/lib/aarch64-linux-gnu/tegra,target=/usr/lib/aarch64-linux-gnu/tegra,readonly \
    set -eux; \
    workdir="$(mktemp -d /tmp/pyds.XXXX)"; \
    cd "$workdir"; \
    git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps; \
    cd deepstream_python_apps; \
    git checkout v1.2.2; \
    git submodule update --init bindings/3rdparty/pybind11 bindings/3rdparty/git-partial-submodule; \
    # JetPack 6.1 ships Python 3.10; adjust upstream defaults and skip analytics enums missing in this image.
    sed -i 's/set(PYTHON_MINOR_VERSION 12)/set(PYTHON_MINOR_VERSION 10)/' bindings/CMakeLists.txt; \
    sed -i 's/set(PYTHON_MINVERS_ALLOWED 12)/set(PYTHON_MINVERS_ALLOWED 10 12)/' bindings/CMakeLists.txt; \
    sed -i '/NVDS_FRAME_META_NVDSANALYTICS/d' bindings/src/bindnvdsmeta.cpp; \
    sed -i '/NVDS_OBJ_META_NVDSANALYTICS/d' bindings/src/bindnvdsmeta.cpp; \
    cd bindings; \
    ls -l $DS_PATH/lib/libnvbufsurface.so; \
    mkdir -p build && cd build; \
    cmake .. -DPYTHON_MAJOR_VERSION=3 -DPYTHON_MINOR_VERSION=10 -DDS_PATH=$DS_PATH; \
    make -j"$(nproc)"; \
    python3 -m pip install --no-cache-dir .; \
    cd /; rm -rf "$workdir"

COPY docker/deepstream/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /app

ENTRYPOINT ["/entrypoint.sh"]
