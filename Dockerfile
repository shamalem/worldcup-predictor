# syntax=docker/dockerfile:1
#
# Single-image build for Render (or any Docker host).
# Stage 1 builds the React/Vite frontend; stage 2 runs the FastAPI backend and
# serves the built frontend from the same origin, so there are no CORS issues.
# The model is downloaded + trained during the build, so the running container
# starts instantly with the artifacts baked in.

# ---------- Stage 1: build the frontend ----------
FROM node:20-slim AS frontend
WORKDIR /fe
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build          # outputs /fe/dist

# ---------- Stage 2: backend + static site ----------
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# libgomp1 is required by XGBoost.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching).
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install -r backend/requirements.txt

# Copy backend source and the built frontend.
COPY backend/ ./backend/
COPY --from=frontend /fe/dist ./backend/app/static

# Download the dataset and train both models at build time so the image ships
# ready to serve. (Needs network during build — Render provides it.)
WORKDIR /app/backend
RUN python -m scripts.download_data && python -m ml.train

# Render injects $PORT; default to 8000 for plain `docker run`.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
