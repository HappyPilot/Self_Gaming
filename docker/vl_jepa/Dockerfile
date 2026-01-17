FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cusparselt/lib:/usr/lib/aarch64-linux-gnu/nvidia:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}

ARG TORCH_WHEEL_URL="https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+3bcc3cddb5.nv24.07.16234504-cp310-cp310-linux_aarch64.whl"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -U pip && \
    python3 -m pip install --no-cache-dir "${TORCH_WHEEL_URL}" && \
    python3 -m pip install --no-cache-dir \
        "numpy<2" \
        nvidia-cusparselt-cu12 \
        paho-mqtt \
        pillow \
        pycuda \
        transformers

WORKDIR /app

ENTRYPOINT ["python3", "/app/vl_jepa_agent.py"]
