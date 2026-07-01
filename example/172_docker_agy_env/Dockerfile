# =====================================================================
# Stage 1: Build Environment & Cache Dependency Compilation
# =====================================================================
FROM ubuntu:26.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Grab the modern uv binary directly from Astral's official release container
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install base toolsets required to fetch the CLI and compile Python extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 1. Install the programmatic Python SDK using uv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv .venv && \
    uv pip install --no-cache-dir google-antigravity

# 2. Safely download, decompress, and run the installation script
RUN curl -fsSL --compressed https://antigravity.google/cli/install.sh -o install.sh && \
    chmod +x install.sh && \
    ./install.sh


# =====================================================================
# Stage 2: Insulated Runtime Runner
# =====================================================================
FROM ubuntu:26.04 AS runner

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Install the essential tool belt for agy agents (Git, Curl, basic utilities)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-minimal \
    git \
    curl \
    ca-certificates \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Copy the isolated Python environment
COPY --from=builder /workspace/.venv /workspace/.venv

# Copy the actual agy CLI tool from the builder's local path to system binaries
COPY --from=builder /root/.local/bin/agy /usr/local/bin/agy

# Add the Antigravity clean environment tweaks to .bashrc to prevent interactive loop freezing
RUN echo 'if [[ -n "$ANTIGRAVITY_AGENT" ]]; then export TERM=dumb; export DEBIAN_FRONTEND=noninteractive; unalias -a; export PS1="\$ "; fi' >> /root/.bashrc

# Prepend the virtual environment's bin folder to the execution PATH
ENV PATH="/workspace/.venv/bin:$PATH"

VOLUME [ "/workspace/src" ]

# Default to launching a bash shell where 'agy' works out of the box
CMD ["/bin/bash"]
