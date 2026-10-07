# Tesseract 5 training tools plus tesstrain, used by train.sh.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr libtesseract-dev libleptonica-dev \
        make git wget ca-certificates bc python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/tesseract-ocr/tesstrain /tesstrain
RUN python3 -m venv /venv && /venv/bin/pip install --no-cache-dir -r /tesstrain/requirements.txt
ENV PATH="/venv/bin:${PATH}"
WORKDIR /tesstrain
