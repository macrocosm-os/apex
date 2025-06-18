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

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . ./gpu_container/

# Command to run the application
CMD ["uvicorn", "gpu_container.app:app", "--host", "0.0.0.0", "--port", "8000"]
