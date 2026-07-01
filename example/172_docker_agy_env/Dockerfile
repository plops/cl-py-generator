# =====================================================================
# Stage 1: Build & Dependency Gathering
# =====================================================================
FROM ubuntu:26.04 AS builder

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install minimal system requirements for Python and dependency compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    python3-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the modern uv binary directly from Astral's official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency definitions (pyproject.toml or requirements.txt)
COPY requirements.txt .

# Use uv to compile and install dependencies straight into a virtual environment
RUN uv venv .venv && \
    .venv/bin/pip install --no-cache-dir -r requirements.txt

# =====================================================================
# Stage 2: Final Slim Runtime Image
# =====================================================================
FROM ubuntu:26.04 AS runner

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install ONLY the bare-minimum Python runtime (no compilers, no uv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-minimal \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the isolated Python environment built by uv from Stage 1
COPY --from=builder /app/.venv /app/.venv
COPY . /app

# Place the virtual environment's executable pathway onto the system PATH
ENV PATH="/app/.venv/bin:$PATH"

# Run your Python-based Antigravity Agent script
CMD ["python", "agent.py"]
