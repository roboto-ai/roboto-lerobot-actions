ARG PYTHON_MAJOR=3
ARG PYTHON_MINOR=12
ARG OS_VARIANT=slim-bookworm
FROM --platform=linux/amd64 public.ecr.aws/docker/library/python:${PYTHON_MAJOR}.${PYTHON_MINOR}-${OS_VARIANT}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        pkg-config \
        linux-headers-generic \
        ffmpeg \
        libavformat-dev \
        libavcodec-dev \
        libavdevice-dev \
        libavutil-dev \
        libswscale-dev \
        libswresample-dev \
        libavfilter-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.runtime.txt ./
RUN python -m pip install --upgrade pip setuptools && python -m pip install -r requirements.runtime.txt

COPY src/enrich_lerobot_dataset/ ./enrich_lerobot_dataset

# Force the stdout and stderr streams to be unbuffered to decrease latency when viewing real-time action logs
# and to maximum context captured in logs if this action fails.
ENV PYTHONUNBUFFERED=0

# Pass args to your script at runtime by setting the CMD.
# Those args can be parsed by argparse, for example.
ENTRYPOINT [ "python", "-m", "enrich_lerobot_dataset" ]
