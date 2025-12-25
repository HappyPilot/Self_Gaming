FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends procps python3-dev build-essential libffi-dev; \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir --ignore-installed \
      paho-mqtt requests flask \
      titans-pytorch \
      transformers==4.38.2 timm==0.9.16 opencv-python-headless==4.9.0.80 huggingface-hub==0.24.6

RUN find /usr/lib/aarch64-linux-gnu/nvidia/weston -type f -size 0 -delete || true

COPY docker/health/policy.sh /usr/local/bin/policy-healthcheck
RUN chmod +x /usr/local/bin/policy-healthcheck

WORKDIR /app

ENTRYPOINT ["python3", "/app/policy_agent.py"]
