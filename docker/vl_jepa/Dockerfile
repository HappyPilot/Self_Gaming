FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -U pip && \
    python3 -m pip install --no-cache-dir \
        "numpy<2" \
        paho-mqtt \
        pillow \
        pycuda \
        transformers

WORKDIR /app

ENTRYPOINT ["python3", "/app/vl_jepa_agent.py"]
