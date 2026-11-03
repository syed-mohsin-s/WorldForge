# syntax=docker/dockerfile:1.7
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

COPY frontend/package.json frontend/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm,sharing=locked \
    npm ci --cache=/root/.npm --prefer-offline

COPY frontend/ ./
RUN npm run build

# -----------------------------------------------------------------------------
# Stage 2: Python 3.12 + MuJoCo 3.17 Runtime (Powered by Astral uv)
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

# Copy Astral uv binary (10-100x faster than pip, multiplexed parallel downloads)
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/

WORKDIR /app

# Default MUJOCO_GL to egl (with automatic fallback to osmesa in base_env.py)
# to prevent OSMesa + PyTorch thread/TLS segfaults during SB3 PPO training.
ENV DEBIAN_FRONTEND=noninteractive \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_TORCH_BACKEND=cpu \
    UV_HTTP_TIMEOUT=120 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MUJOCO_GL=egl \
    PORT=8000

# Install system dependencies for MuJoCo physics, headless rendering & OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libegl1 \
    libegl-mesa0 \
    libgbm1 \
    libgles2 \
    libosmesa6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# The lock contains the complete, CPU-only Linux dependency graph.
COPY requirements.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip sync --system --torch-backend cpu --no-progress requirements.lock

# Copy Python codebase, configs, and scripts (outputs/ is created dynamically
# at runtime rather than copied from the build context, so fresh clones succeed).
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Ensure output directories exist with write permissions
RUN mkdir -p /app/outputs/uploads /app/outputs/runs

# Copy compiled frontend from Stage 1 into /app/frontend/dist
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Health check against FastAPI backend
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

EXPOSE 8000

# Launch WorldForge FastAPI Backend (which also serves the compiled React Studio at '/')
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
