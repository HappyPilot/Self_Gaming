# DeepStream-based perception image (JetPack 6.1 / DeepStream 7.1)
FROM nvcr.io/nvidia/deepstream-l4t:7.1-triton-multiarch

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        python3-pip python3-gi python3-dev \
        git cmake build-essential libglib2.0-dev pkg-config; \
    pip3 install --no-cache-dir paho-mqtt cython

COPY docker/deepstream/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /app

ENTRYPOINT ["/entrypoint.sh"]
