FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        procps \
        python3-numpy; \
    apt-get purge -y python3-blinker python3-typing-extensions python3-pyparsing || true; \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
        "numpy<2" \
        paho-mqtt \
        easyocr==1.7.1 \
        opencv-python \
        onnxruntime==1.16.3 \
        rapidocr-onnxruntime==1.3.5 \
    && pip3 uninstall -y opencv-python-headless || true

RUN find /usr/lib/aarch64-linux-gnu/nvidia/weston -type f -size 0 -delete || true

COPY docker/health/ocr_easy.sh /usr/local/bin/ocr-easy-healthcheck
RUN chmod +x /usr/local/bin/ocr-easy-healthcheck

WORKDIR /app

ENTRYPOINT ["python3", "/app/ocr_easy_agent.py"]
