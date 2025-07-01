FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Set the working directory
WORKDIR /app

# Install Python 3.9
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3.9 python3.9-dev python3.9-distutils curl && \
    # Install pip for python3.9
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9 && \
    # Make python3 point to python3.9
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    # Clean up
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Build args
ARG MODEL_ID
ARG VLLM_MODEL_ID
ARG VLLM_GPU_UTILIZATION
ARG HF_MODEL_PATH
ARG BLOCK_STARTUP

# Environment variables
ENV MODEL_ID=${MODEL_ID}
ENV VLLM_MODEL_ID=${VLLM_MODEL_ID}
ENV VLLM_GPU_UTILIZATION=${VLLM_GPU_UTILIZATION}
ENV HF_MODEL_PATH=${HF_MODEL_PATH}
ENV BLOCK_STARTUP=${BLOCK_STARTUP}
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Install model downloader deps
RUN pip install --no-cache-dir huggingface-hub

# Copy and run model downloader
COPY download_model.py .
RUN python3 download_model.py --model-name "$VLLM_MODEL_ID" --model-path "$HF_MODEL_PATH"

# Copy the requirements file into the container
COPY --from=project_context /gpu_container/requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY --from=project_context /gpu_container/ ./gpu_container/

# Command to run the application
CMD ["uvicorn", "gpu_container.app:app", "--host", "0.0.0.0", "--port", "8000"]
