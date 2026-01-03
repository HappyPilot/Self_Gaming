FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        procps; \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
        "numpy<2" \
        paho-mqtt \
        pillow

WORKDIR /app

ENTRYPOINT ["python3", "/app/vl_jepa_agent.py"]
