# =============================================================================
# WorldForge-RL Multi-Stage Production Container
# MuJoCo 3.17 Physics + Stable-Baselines3 + FastAPI + React Studio Frontend
# Powered by Astral uv for ultra-fast, resilient package resolution
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build React/Vite/Tailwind Frontend
# -----------------------------------------------------------------------------
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci || npm install --legacy-peer-deps

COPY frontend/ ./
RUN npm run build

# -----------------------------------------------------------------------------
# Stage 2: Python 3.12 + MuJoCo 3.17 Runtime (Powered by Astral uv)
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

# Copy Astral uv binary (10-100x faster than pip, multiplexed parallel downloads)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_TORCH_BACKEND=cpu \
    UV_HTTP_TIMEOUT=120 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MUJOCO_GL=osmesa \
    PORT=8000

# Install system dependencies for MuJoCo physics, headless rendering & OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libegl1 \
    libgles2 \
    libosmesa6 \
    libosmesa6-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install all Python dependencies using uv with BuildKit cache mount and native CPU torch backend
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements.txt

# Copy Python codebase, configs, scripts, and pre-rendered preview assets
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY outputs/ ./outputs/

# Ensure output directories exist with write permissions
RUN mkdir -p /app/outputs/uploads /app/outputs/runs

# Copy compiled frontend from Stage 1 into /app/frontend/dist
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Health check against FastAPI backend
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

EXPOSE 8000

# Launch WorldForge FastAPI Backend (which also serves the compiled React Studio at '/')
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
