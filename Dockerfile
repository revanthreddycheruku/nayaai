FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV HOST=0.0.0.0
ENV CHROMA_PATH=/app/vector_store
ENV ALLOW_UNVERIFIED_CONTEXT=false
ENV HOME=/home/user
ENV TRANSFORMERS_CACHE=/home/user/.cache/huggingface
ENV TORCH_HOME=/home/user/.cache/torch

WORKDIR /app

# Install system packages & Node.js 20
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    git \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create standard non-root user (UID 1000) for Hugging Face Spaces
RUN useradd -m -u 1000 user && \
    mkdir -p /home/user/.cache/huggingface /home/user/.cache/torch && \
    chown -R user:user /home/user

# Install Python requirements
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r ./backend/requirements.txt

# Install frontend dependencies and build production assets
COPY frontend/package*.json ./frontend/
RUN cd frontend && npm install
COPY frontend ./frontend
RUN cd frontend && npm run build

# Copy statutes, data, scripts, and backend code
COPY data ./data
COPY scripts ./scripts
COPY backend ./backend
COPY vector_store ./vector_store

# Set permissions for non-root user
RUN chown -R user:user /app

USER user

EXPOSE 7860

# Launch unified Uvicorn server on port 7860
CMD ["sh", "-c", "uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port ${PORT:-7860}"]
