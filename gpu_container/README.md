# GPU Container

This directory contains the necessary files to build and run a Docker container for the GPU-accelerated parts of this project.

## Prerequisites

- [Docker](https.docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Building the Container

To build the Docker image, navigate to this directory (`gpu_container`) and run the following command:

```bash
docker build -t gpu-app .
```

## Running the Container

To run the container and verify that the GPU is accessible, use the following command:

```bash
docker run --gpus all gpu-app
```

You should see output indicating that CUDA is available and listing your GPU details.

## Using Docker Compose (Recommended)

To make things even simpler, you can use the provided `docker-compose.yml` file. This file automates the build process and the GPU resource allocation.

From this directory, run the following command to build and start the container:

```bash
docker compose up --build
```

The container will start, and you will see the output from `app.py` directly in your terminal. To stop the container, press `Ctrl+C`.

## Running your application
You will need to modify the `Dockerfile` to copy your application code and run it. The current `app.py` is a placeholder.
To run your actual application, you might want to mount your project directory into the container. For example:

```bash
docker run --gpus all -v /path/to/your/project:/app my-gpu-app
```

Remember to replace `/path/to/your/project` with the actual path to your project on your host machine.
You might also need to adjust the `CMD` in the `Dockerfile` to run your specific application entry point. 