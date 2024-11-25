# syntax=docker/dockerfile:1

FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04 AS base

# Define build arguments
ARG USER=vscode
ARG USER_UID=1000
ARG USER_GID=1000
ARG DEVTOOLS_DIR=/usr/local/bin/

# Consolidate environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    CUDA_HOME=/usr/local/cuda \
    UV_LINK_MODE=copy \
    PATH=/usr/local/cuda/bin:/opt/bin:/home/${USER}/.local/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib:${LD_LIBRARY_PATH}

# Set shell to bash with strict mode
SHELL ["/bin/bash", "-o", "errexit", "-o", "nounset", "-o", "pipefail", "-c"]

# Create the user and install essential packages in a single RUN command to reduce layers
RUN groupadd --gid $USER_GID $USER \
    && useradd --uid $USER_UID --gid $USER_GID -m $USER \
    && apt-get update && apt-get install -y --no-install-recommends \
        sudo \
        ca-certificates \
        curl \
        git \
        zsh \
        unzip \
        rustc \
        cargo \
        shellcheck \
        shfmt \
        jq \
        fzf \
        bat \
        ripgrep \
    && echo "$USER ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USER \
    && chmod 0440 /etc/sudoers.d/$USER \
    # Clean up APT cache to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Multi-stage build for installing uv
FROM base AS uv-installer

# Install uv as root (default user)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mkdir -p /opt/bin/ \
    && mv /root/.local/bin/uv /opt/bin/ \
    && mv /root/.local/bin/uvx /opt/bin/

# Final stage
FROM base

# Copy uv binaries from uv-installer stage
COPY --from=uv-installer /opt/bin/uv /usr/local/bin/
COPY --from=uv-installer /opt/bin/uvx /usr/local/bin/

# Use solzard images to copy additional tools
COPY --from=solzard/xh:latest --chown=${USER}:${USER} /bin/xh ${DEVTOOLS_DIR}
COPY --from=solzard/fzf:latest --chown=${USER}:${USER} /bin/fzf ${DEVTOOLS_DIR}
COPY --from=solzard/lefthook:latest --chown=${USER}:${USER} /bin/lefthook ${DEVTOOLS_DIR}
COPY --from=solzard/bat:latest --chown=${USER}:${USER} /bin/bat ${DEVTOOLS_DIR}
COPY --from=solzard/eza:latest --chown=${USER}:${USER} /bin/eza ${DEVTOOLS_DIR}
COPY --from=solzard/fd:latest --chown=${USER}:${USER} /bin/fd ${DEVTOOLS_DIR}
COPY --from=solzard/delta:latest --chown=${USER}:${USER} /bin/delta ${DEVTOOLS_DIR}
COPY --from=solzard/htmlq:latest --chown=${USER}:${USER} /bin/htmlq ${DEVTOOLS_DIR}
COPY --from=solzard/procs:latest --chown=${USER}:${USER} /bin/procs ${DEVTOOLS_DIR}
COPY --from=solzard/rg:latest --chown=${USER}:${USER} /bin/rg ${DEVTOOLS_DIR}
COPY --from=solzard/sd:latest --chown=${USER}:${USER} /bin/sd ${DEVTOOLS_DIR}
COPY --from=solzard/tokei:latest --chown=${USER}:${USER} /bin/tokei ${DEVTOOLS_DIR}
COPY --from=solzard/xh:latest --chown=${USER}:${USER} /bin/xh ${DEVTOOLS_DIR}

# Switch to non-root user
USER $USER

# Set working directory
WORKDIR /workspace

# Copy the entire project into the Docker image
COPY --chown=${USER}:${USER} . /workspace/

# Grant execute permissions to post_create.sh script
RUN chmod +x /workspace/.devcontainer/post_create.sh

# Ensure uv is in the PATH
ENV PATH="/home/${USER}/.local/bin:${PATH}"

# Create a virtual environment using uv
RUN uv venv

# Set default shell to zsh
SHELL ["/bin/zsh", "-c"]

# Set the default command to keep the container running interactively
CMD ["tail", "-f", "/dev/null"]
