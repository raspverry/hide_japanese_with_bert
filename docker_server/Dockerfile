# syntax=docker/dockerfile:1

# Builder stage
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04 AS builder

# Define build arguments
ARG USER=vscode
ARG USER_UID=1000
ARG USER_GID=1000

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/opt/bin:/home/${USER}/.local/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib:${LD_LIBRARY_PATH}

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mkdir -p /opt/bin/ \
    && mv /root/.local/bin/uv /opt/bin/ \
    && mv /root/.local/bin/uvx /opt/bin/

# Final stage
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# Copy arguments from builder
ARG USER=vscode
ARG USER_UID=1000
ARG USER_GID=1000

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    UV_LINK_MODE=copy \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/opt/bin:/home/${USER}/.local/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib:${LD_LIBRARY_PATH} \
    SHELL=/bin/zsh \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install runtime dependencies including zsh
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    zsh \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    # Create user
    && groupadd --gid $USER_GID $USER \
    && useradd --uid $USER_UID --gid $USER_GID -m $USER \
    && echo "$USER ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USER \
    && chmod 0440 /etc/sudoers.d/$USER \
    # Change default shell to zsh
    && chsh -s /bin/zsh $USER

# Install Oh My Zsh and configure it
USER $USER
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
    && git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
    && git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting \
    && sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions zsh-syntax-highlighting)/' ~/.zshrc

# Copy uv from builder stage
COPY --from=builder /opt/bin/uv /usr/local/bin/
COPY --from=builder /opt/bin/uvx /usr/local/bin/

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY --chown=${USER}:${USER} . /workspace/

# Create venv and install dependencies
RUN uv venv \
    && . .venv/bin/activate \
    && uv sync

# Copy the rest of the application
COPY --chown=${USER}:${USER} . .

# Grant execute permissions to scripts
RUN chmod +x /workspace/.devcontainer/post_create.sh

# Expose port
EXPOSE 8000

# Use zsh as the default shell for RUN commands
SHELL ["/bin/zsh", "-c"]

# Start with an interactive shell
CMD ["/bin/zsh"]

# Set entrypoint
# ENTRYPOINT ["/workspace/.venv/bin/uvicorn"]
# CMD ["server:app", "--host", "0.0.0.0", "--port", "8000"]