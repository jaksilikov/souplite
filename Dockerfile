# SoupLite Dockerfile - Low-RAM (< 2GB RAM) Optimized
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Environment variables for minimal memory consumption
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
    MALLOC_TRIM_THRESHOLD_=100000 \
    PYTHONMALLOC=malloc \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

# Install Python & system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /workspace

# Install local package in editable or standard mode
COPY . /workspace/
RUN pip install --no-cache-dir -e .[train,serve,data,eval]

ENTRYPOINT ["souplite"]
CMD ["--help"]
