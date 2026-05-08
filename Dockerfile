FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.6;9.0+PTX"
ENV CMAKE_CUDA_ARCHITECTURES="86;90"

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        "numpy<1.24" \
        "protobuf<3.20" \
        tensorflow==2.11.0 \
        waymo-open-dataset-tf-2-11-0 \
        shapely \
        matplotlib \
        tqdm \
        pandas \
        scipy

WORKDIR /workspace/GameFormer

CMD ["bash"]
