FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Set the working directory
WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y python3-pip

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Command to run the application
CMD ["uvicorn", "gpu_container.app:app", "--host", "0.0.0.0", "--port", "8000"] 