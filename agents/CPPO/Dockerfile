FROM python:3.11-slim AS core

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    pkg-config \
    sudo \
    vim \
    nano \
    curl \
    wget \
    unzip \
    tar \
    zip \
    gzip \
    bzip2 \
    procps \
    libhdf5-dev \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN useradd -m -u 1000 -s /bin/bash app

ENV UV_SYSTEM_PYTHON=1
WORKDIR /home/app/mava

ARG USE_CUDA=false

COPY pyproject.toml .
COPY uv.lock .

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$USE_CUDA" = true ] ; then \
        uv sync --locked --no-install-project --extra cuda12  ; \
    else \
        uv sync --locked --no-install-project ; \
    fi

    COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$USE_CUDA" = true ] ; then \
        uv sync --locked --extra cuda12  ; \
    else \
        uv sync --locked ; \
    fi

RUN uv pip install nvitop

RUN chown -R app:app /home/app

USER app

EXPOSE 6006