# Use the official CUDA 11.7 base image with cuDNN and Ubuntu 22.04
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV PYTHON_VERSION=3.10
ENV PYTHONUNBUFFERED=1

# Install Python and system dependencies, then upgrade pip and install Python packages
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python3-pip \
    python3-dev \
    wget \
    zsh \
    && pip3 install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

RUN pip install cupy-cuda117 \
    && pip install spacy[cuda117]

# Download SpaCy model
RUN python3 -m spacy download en_core_web_trf

# Copy application code
COPY app/ .

# Expose port 8000 (optional, remove if not needed)
EXPOSE 8000

# Set default shell to zsh
SHELL ["/bin/zsh", "-c"]

# Set the default command to keep the container running interactively
CMD ["tail", "-f", "/dev/null"]