# Match host JetPack (R36.4.x) with TRT 8.6.x runtime
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        procps \
        v4l-utils \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        libnvvpi3 \
        libegl1 \
        libv4l-0; \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir paho-mqtt requests

RUN find /usr/lib/aarch64-linux-gnu/nvidia/weston -type f -size 0 -delete || true

COPY docker/health/vision.sh /usr/local/bin/vision-healthcheck
COPY docker/health/cursor.sh /usr/local/bin/cursor-healthcheck
RUN chmod +x /usr/local/bin/vision-healthcheck
RUN chmod +x /usr/local/bin/cursor-healthcheck

WORKDIR /app

ENTRYPOINT ["python3", "/app/vision_agent.py"]
