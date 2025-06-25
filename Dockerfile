###############################################################################
# DreamScene – CUDA 12.1 image
#   • Base:   nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04  :contentReference[oaicite:0]{index=0}
#   • Python: system 3.10 (default on 22.04)                :contentReference[oaicite:1]{index=1}
#   • Torch:  2.2.0 + cu121 wheels from the official index  :contentReference[oaicite:2]{index=2}
###############################################################################
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST="8.9" \
    CUDA_HOME="/usr/local/cuda"

# --------------------------------------------------------------------------- #
# System packages
# --------------------------------------------------------------------------- #
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ninja-build cmake curl ca-certificates gnupg \
        python3 python3-venv python3-dev \
        libgl1-mesa-glx libglm-dev wget && \
    rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------- #
# Virtual-env
# --------------------------------------------------------------------------- #
RUN python3 -m venv /opt/dreamscene
ENV PATH="/opt/dreamscene/bin:${PATH}"
RUN pip install --no-cache-dir -U pip setuptools wheel

# --------------------------------------------------------------------------- #
# Core DL stack – CUDA 12.1 wheels
# --------------------------------------------------------------------------- #
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==2.2.0+cu121 torchvision==0.17.0+cu121 torchaudio==2.2.0+cu121

# --------------------------------------------------------------------------- #
# Clone DreamScene
# --------------------------------------------------------------------------- #
WORKDIR /workspace
RUN git clone --recursive https://github.com/DreamScene-Project/DreamScene.git

# --------------------------------------------------------------------------- #
# Python deps (patch out the cu118 pins first)
# --------------------------------------------------------------------------- #
RUN sed -i -E '/^(torch|torchvision|torchaudio)==/d' DreamScene/requirements.txt && \
    pip install -r DreamScene/requirements.txt                         \
              --extra-index-url https://download.pytorch.org/whl/cu121

# --------------------------------------------------------------------------- #
# Gaussian rasteriser & simple-knn
# --------------------------------------------------------------------------- #
RUN git clone --recursive https://github.com/DreamScene-Project/comp-diff-gaussian-rasterization.git && \
    pip install ./comp-diff-gaussian-rasterization && \
    git clone https://github.com/YixunLiang/simple-knn.git && \
    pip install ./simple-knn

# --------------------------------------------------------------------------- #
# PyTorch3D
# --------------------------------------------------------------------------- #
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# --------------------------------------------------------------------------- #
# Point-E (Cap3D fork)
# --------------------------------------------------------------------------- #
RUN git clone https://github.com/crockwell/Cap3D.git && \
    pip install -e Cap3D/text-to-3D/point-e

# ---- OPTIONAL: pre-download finetuned Point-E weight (~2 GB) -------------- #
ARG FETCH_POINTE_MODEL=0
RUN if [ "$FETCH_POINTE_MODEL" = "1" ]; then \
        mkdir -p DreamScene/point_e_model_cache && \
        wget -q https://huggingface.co/datasets/tiange/Cap3D/resolve/main/misc/our_finetuned_models/pointE_finetuned_with_825kdata.pth \
            -O DreamScene/point_e_model_cache/pointE_finetuned_with_825kdata.pth ; \
    fi

# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
WORKDIR /workspace/DreamScene
ARG HF_TOKEN   # provide with  --build-arg HF_TOKEN=hf_your_token

RUN apt-get update && apt-get install -y libglib2.0-0 && \
    apt-get install -y libsm6 libxext6 libxrender1

RUN huggingface-cli login --token "$HF_TOKEN"

VOLUME ["/workspace/DreamScene/output"]
CMD ["/bin/bash"]
