# .devcontainer/Dockerfile

# syntax=docker/dockerfile:1

FROM debian:bookworm AS uv-installer

ARG USER=vscode
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "errexit", "-o", "nounset", "-o", "pipefail", "-c"]

# hadolint ignore=DL3008
RUN <<EOF
    # Install apt-packages.
    apt_packages_to_install=(ca-certificates curl rustc cargo)
    apt-get update --yes
    apt-get install --yes --no-install-recommends "${apt_packages_to_install[@]}"

    # Install uv.
    curl -LsSf https://astral.sh/uv/install.sh | sh
    mkdir -p /opt/bin/
    mv "$HOME"/.local/bin/* /opt/bin/
EOF

# Base image with necessary tools
FROM mcr.microsoft.com/devcontainers/base:debian AS base

# Define build arguments early
ARG USER=vscode \
    DEVTOOLS_DIR=/usr/local/bin/
ARG USER_UID=1000
ARG USER_GID=1000

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# Install essential packages
RUN apt-get update && apt-get install -y \
    curl \
    git \
    zsh \
    sudo \
    unzip \
    && rm -rf /var/lib/apt/lists/* 

# Install additional tools via apt
RUN apt-get update && apt-get install -y --no-install-recommends \
    shellcheck \
    shfmt \
    jq \
    fzf \
    bat \
    ripgrep \
    && rm -rf /var/lib/apt/lists/*

# Copy binaries from the uv-installer image to the devcontainer.
COPY --from=uv-installer --chown=${USER}:${USER} --link /opt/bin/uv ${DEVTOOLS_DIR}
COPY --from=uv-installer --chown=${USER}:${USER} --link /opt/bin/uvx ${DEVTOOLS_DIR}

COPY --from=solzard/xh:latest --chown=${USER}:${USER} --link /bin/xh ${DEVTOOLS_DIR}

# Copy binaries from unofficial images to the devcontainer.
# @see https://hub.docker.com/r/solzard/fzf
# @see https://github.com/junegunn/fzf
COPY --from=solzard/fzf:latest --chown=${USER}:${USER} --link /bin/fzf ${DEVTOOLS_DIR}
# @see https://hub.docker.com/r/solzard/lefthook
# @see https://github.com/evilmartians/lefthook
COPY --from=solzard/lefthook:latest --chown=${USER}:${USER} --link /bin/lefthook ${DEVTOOLS_DIR}
# @see https://hub.docker.com/r/solzard/bat
# @see https://github.com/sharkdp/bat
COPY --from=solzard/bat:latest --chown=${USER}:${USER} --link /bin/bat ${DEVTOOLS_DIR}
# @see https://hub.docker.com/r/solzard/eza
# @see https://github.com/eza-community/eza
COPY --from=solzard/eza:latest --chown=${USER}:${USER} --link /bin/eza ${DEVTOOLS_DIR}
# @see https://hub.docker.com/r/solzard/fd
# @see https://github.com/sharkdp/fd
COPY --from=solzard/fd:latest --chown=${USER}:${USER} --link /bin/fd ${DEVTOOLS_DIR}
# @see https://hub.docker.com/r/solzard/delta
# @see https://github.com/dandavison/delta
COPY --from=solzard/delta:latest --chown=${USER}:${USER} --link /bin/delta ${DEVTOOLS_DIR}
# @see https://hub.docker.com/r/solzard/htmlq
# @see https://github.com/mgdm/htmlq
COPY --from=solzard/htmlq:latest --chown=${USER}:${USER} --link /bin/htmlq ${DEVTOOLS_DIR}
# @see https://hub.docker.com/r/solzard/procs
# @see https://github.com/dalance/procs
COPY --from=solzard/procs:latest --chown=${USER}:${USER} --link /bin/procs ${DEVTOOLS_DIR}
# @see https://hub.docker.com/r/solzard/rg
# @see https://github.com/BurntSushi/ripgrep
COPY --from=solzard/rg:latest --chown=${USER}:${USER} --link /bin/rg ${DEVTOOLS_DIR}
# @see https://hub.docker.com/r/solzard/sd
# @see https://github.com/chmln/sd
COPY --from=solzard/sd:latest --chown=${USER}:${USER} --link /bin/sd ${DEVTOOLS_DIR}
# @see https://hub.docker.com/r/solzard/tokei
# @see https://github.com/XAMPPRocky/tokei
COPY --from=solzard/tokei:latest --chown=${USER}:${USER} --link /bin/tokei ${DEVTOOLS_DIR}
# @see https://hub.docker.com/r/solzard/xh
# @see https://github.com/ducaale/xh
COPY --from=solzard/xh:latest --chown=${USER}:${USER} --link /bin/xh ${DEVTOOLS_DIR}

# Switch to non-root user
USER vscode

# Set work directory
WORKDIR /workspace

# Copy the entire project into the Docker image
COPY --chown=vscode:vscode . /workspace/

RUN sudo chown -R ${USER}:${USER} /workspace

# Set execute permissions for post_create.sh
RUN chmod +x /workspace/.devcontainer/post_create.sh

# Install uv
# RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Ensure uv is in the PATH
ENV PATH="/home/${USER}/.local/bin:${PATH}"

# RUN which uv && ls -l /workspace/${USER}/.local/bin/uv
# Create a virtual environment using uv
RUN export PATH="/home/${USER}/.local/bin:${PATH}" && uv venv

# Activate the virtual environment in ~/.bashrc and ~/.zshrc
# RUN echo 'source /workspace/.venv/bin/activate' >> ~/.bashrc \
#     && echo 'source /workspace/.venv/bin/activate' >> ~/.zshrc

# # Set environment variable for UV_LINK_MODE
ENV UV_LINK_MODE=copy

# # Copy pyproject.toml and install dependencies
# COPY --chown=vscode:vscode pyproject.toml /workspace/
RUN . /workspace/.venv/bin/activate && uv pip install -e .

# add ruff
RUN curl -LsSf https://astral.sh/ruff/install.sh | sh

RUN uv add spacy && \
    uv pip install pip && \
    uv run python -m spacy download ja_core_news_lg

# Set default shell to zsh
SHELL ["/bin/zsh", "-c"]

# **Set the default command to keep the container running interactively**
CMD ["tail", "-f", "/dev/null"]
