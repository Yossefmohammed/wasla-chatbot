# =========================
# 1. Use slim Python image
# =========================
FROM python:3.10-slim

# =========================
# 2. Set environment variables
# =========================
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
    TOKENIZERS_PARALLELISM=false

# =========================
# 3. Install system dependencies
# =========================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# =========================
# 4. Install Python packages (no cache)
# =========================
RUN pip install --upgrade pip --no-cache-dir

# Only install the packages you actually need:
RUN pip install --no-cache-dir \
    streamlit \
    torch \
    transformers \
    bitsandbytes \
    sentence-transformers \
    chromadb \
    langchain \
    langchain-community

# =========================
# 5. Copy your app
# =========================
WORKDIR /app
COPY . .

# =========================
# 6. Expose port for Streamlit
# =========================
EXPOSE 8501

# =========================
# 7. Run Streamlit
# =========================
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
