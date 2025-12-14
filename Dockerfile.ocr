FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        procps \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-rus; \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir pytesseract pillow paho-mqtt

RUN find /usr/lib/aarch64-linux-gnu/nvidia/weston -type f -size 0 -delete || true

COPY docker/health/ocr.sh /usr/local/bin/ocr-healthcheck
RUN chmod +x /usr/local/bin/ocr-healthcheck

WORKDIR /app

ENTRYPOINT ["python3", "/app/ocr_agent.py"]
