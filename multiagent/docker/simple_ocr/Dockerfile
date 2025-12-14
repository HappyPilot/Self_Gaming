FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends tesseract-ocr libtesseract-dev libleptonica-dev python3-pip; \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir pytesseract pillow paho-mqtt numpy

WORKDIR /app

ENTRYPOINT ["python3", "/app/simple_ocr_agent.py"]
