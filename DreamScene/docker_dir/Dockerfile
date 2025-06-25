# ---------- base image --------------------------------------------------------
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

# ---------- system deps -------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl wget ca-certificates gnupg2 \
    python3.10 python3.10-venv python3-pip \
    libgl1-mesa-glx ninja-build && \
    rm -rf /var/lib/apt/lists/*

# ---------- python env --------------------------------------------------------
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" TORCH_CUDA_ARCH_LIST="8.9" CUDA_HOME="/usr/local/cuda"

RUN pip install --upgrade pip wheel setuptools "numpy<2"

# Torch 2.2 + cu121 wheels
RUN pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 torchaudio==2.2.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# ---------- third-party libs --------------------------------------------------
WORKDIR /opt/src

# simple-knn
# RUN git clone https://github.com/YixunLiang/simple-knn.git && \
#     pip install --no-build-isolation -v -e simple-knn

# rasterizer
# RUN git clone --recursive https://github.com/DreamScene-Project/comp-diff-gaussian-rasterization.git && \
#     pip install --no-build-isolation -e comp-diff-gaussian-rasterization

# PyTorch3D wheel (community)
RUN pip install --extra-index-url https://miropsota.github.io/torch_packages_builder \
    pytorch3d==0.7.5+pt2.2.0cu121

# Point-E fork
RUN git clone https://github.com/crockwell/Cap3D.git && \
    pip install --no-build-isolation -e Cap3D/text-to-3D/point-e

# DreamScene itself
# RUN git clone https://github.com/DreamScene-Project/DreamScene.git && \
#     pip install --no-build-isolation -r DreamScene/requirements.txt && \
#     pip install --no-build-isolation -e DreamScene

# ---------- runtime stage (slimmer) ------------------------------------------
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
ENV PATH="/opt/venv/bin:$PATH" \
    TORCH_CUDA_ARCH_LIST="8.9" \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/src /opt/src
WORKDIR /opt/src/DreamScene

ENTRYPOINT ["python", "-m", "DreamScene.main"]
CMD ["--help"]
