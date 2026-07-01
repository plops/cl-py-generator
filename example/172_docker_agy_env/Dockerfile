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

# Copy the modern uv binary directly from Astral's official release container
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install the essential tool belt for agy agents (full Python, GCC compilation tools, Git, Curl, SBCL, Emacs, rlwrap)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-full \
    build-essential \
    gcc \
    git \
    curl \
    ca-certificates \
    jq \
    sbcl \
    emacs-nox \
    rlwrap \
    && rm -rf /var/lib/apt/lists/*

# Copy the isolated Python environment
COPY --from=builder /workspace/.venv /workspace/.venv

# Copy the actual agy CLI tool from the builder's local path to system binaries
COPY --from=builder /root/.local/bin/agy /usr/local/bin/agy

# Download and install Quicklisp
RUN curl -O https://beta.quicklisp.org/quicklisp.lisp && \
    sbcl --non-interactive --load quicklisp.lisp --eval "(quicklisp-quickstart:install)" && \
    rm quicklisp.lisp

# Append Quicklisp configuration to .sbclrc
RUN echo '#-quicklisp' >> /root/.sbclrc && \
    echo '(let ((quicklisp-init (merge-pathnames "quicklisp/setup.lisp" (user-homedir-pathname))))' >> /root/.sbclrc && \
    echo '  (when (probe-file quicklisp-init)' >> /root/.sbclrc && \
    echo '    (load quicklisp-init)))' >> /root/.sbclrc

# Pre-create local-projects symlinks (which resolve dynamically when /workspace/src is mounted)
RUN mkdir -p /root/quicklisp/local-projects && \
    ln -s /workspace/src/cl-py-generator /root/quicklisp/local-projects/cl-py-generator && \
    ln -s /workspace/src/cl-cpp-generator2 /root/quicklisp/local-projects/cl-cpp-generator2 && \
    ln -s /workspace/src/cl-rust-generator /root/quicklisp/local-projects/cl-rust-generator

# Pre-fetch and cache Quicklisp systems and common dependencies
RUN sbcl --non-interactive \
    --load /root/quicklisp/setup.lisp \
    --eval '(ql:quickload "quicklisp-slime-helper")' \
    --eval '(ql:quickload "alexandria")' \
    --eval '(ql:quickload "jonathan")' \
    --eval '(ql:quickload "external-program")' \
    --eval '(ql:quickload "cl-ppcre")'

# Pre-install Emacs packages
RUN emacs --batch \
    --eval "(require 'package)" \
    --eval "(add-to-list 'package-archives '(\"melpa\" . \"https://melpa.org/packages/\"))" \
    --eval "(package-initialize)" \
    --eval "(package-refresh-contents)" \
    --eval "(dolist (pkg '(cmake-mode company gptel magit markdown-mode orderless paredit slime yaml-mode use-package)) (package-install pkg))"

# Copy the modified .emacs configuration from the build context
COPY .emacs /root/.emacs

# Add the Antigravity clean environment tweaks and dangerous CLI alias to .bashrc
RUN echo 'if [[ -n "$ANTIGRAVITY_AGENT" ]]; then export TERM=dumb; export DEBIAN_FRONTEND=noninteractive; unalias -a; export PS1="\$ "; fi' >> /root/.bashrc && \
    echo "alias agy='agy --dangerously-skip-permissions'" >> /root/.bashrc

# Prepend the virtual environment's bin folder to the execution PATH
ENV PATH="/workspace/.venv/bin:$PATH"

VOLUME [ "/workspace/src" ]

# Default to launching a bash shell where 'agy' works out of the box
CMD ["/bin/bash"]
