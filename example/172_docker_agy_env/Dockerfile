# =====================================================================
# Stage 1: Build Environment & Cache Dependency Compilation
# =====================================================================
FROM ubuntu:26.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Grab the modern uv binary directly from Astral's official release container
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install base toolsets required to compile Python dependencies or wheels 
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install the Antigravity SDK and its dependencies directly into a virtual env
# Leveraging Docker build cache mounts speeds up future rebuilds
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv .venv && \
    uv pip install --no-cache-dir google-antigravity

# =====================================================================
# Stage 2: Insulated Runtime Runner
# =====================================================================
FROM ubuntu:26.04 AS runner

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Install the essential tool belt for agy agents (Git, Curl, basic compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-minimal \
    git \
    curl \
    ca-certificates \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Copy the isolated, complete Python virtual environment from the builder
COPY --from=builder /workspace/.venv /workspace/.venv

# Prepend the virtual environment's bin folder to the execution PATH
ENV PATH="/workspace/.venv/bin:$PATH"

# Expose a default local directory path for mounting code repositories
VOLUME [ "/workspace/src" ]

# Default to launching a bash shell where 'agy' or your scripts can run safely
CMD ["/bin/bash"]