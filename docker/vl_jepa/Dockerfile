FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cusparselt/lib:/usr/lib/aarch64-linux-gnu/nvidia:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}

ARG TORCH_WHL_URL="https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl"
ARG TORCHVISION_WHL_URL="https://pypi.jetson-ai-lab.io/jp6/cu126/+f/907/c4c1933789645/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    libopenblas0 \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -U pip setuptools wheel && \
    python3 -m pip install --no-cache-dir \
        "${TORCH_WHL_URL}" \
        "${TORCHVISION_WHL_URL}" && \
    python3 -m pip install --no-cache-dir \
        "numpy<2" \
        nvidia-cusparselt-cu12 \
        paho-mqtt \
        pillow \
        pycuda \
        sentencepiece \
        protobuf \
        "transformers==4.48.3"

WORKDIR /app

ENTRYPOINT ["python3", "/app/vl_jepa_agent.py"]
