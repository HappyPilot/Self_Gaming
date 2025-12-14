FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        python3-pip \
        python3-numpy \
        python3-opencv \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6; \
    rm -rf /var/lib/apt/lists/*

# Install runtime deps but keep Jetson-provided numpy/opencv to avoid ABI mismatch
RUN pip3 install --no-cache-dir \
    paho-mqtt \
    requests \
    "ultralytics==8.3.226" --no-deps

WORKDIR /app

ENTRYPOINT ["python3", "/app/object_detection_agent.py"]
