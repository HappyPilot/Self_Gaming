FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

# Minimal system deps
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends git libgl1; \
    rm -rf /var/lib/apt/lists/*

# Fix blinker/distutils
RUN pip3 install --break-system-packages --no-cache-dir --ignore-installed "blinker>=1.6"

# Install ONLY communication deps. 
# Ultralytics deps (numpy, torch, cv2, etc) are provided by the base image.
# We skip optional deps (thop, seaborn) to avoid pip pulling in a CPU-only torch.
RUN pip3 install --break-system-packages --no-cache-dir \
    paho-mqtt \
    requests \
    psutil \
    py-cpuinfo

# Clone Ultralytics into a path unaffected by bind mounts
RUN git clone https://github.com/ultralytics/ultralytics /opt/ultralytics_repo

COPY docker/health/vision.sh /usr/local/bin/vision-healthcheck
RUN chmod +x /usr/local/bin/vision-healthcheck

# FORCE COPY AGENT CODE (Bypass volume issues)
COPY agents/object_detection_agent.py /app/
COPY agents/utils /app/utils

WORKDIR /app
ENV PYTHONPATH=/app:/opt/ultralytics_repo
