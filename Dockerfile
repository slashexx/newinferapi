FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .

# Copy LoRA adapter weights
COPY qwen_1_8b_lora/ /app/qwen_1_8b_lora/

# Set environment variables with defaults
ENV MODEL_NAME="Qwen/Qwen-1_8B"
ENV LORA_ADAPTER_PATH="/app/qwen_1_8b_lora"
ENV PORT=9000
ENV MAX_MODEL_LEN=8192
ENV API_KEY=""

# Create cache directories
RUN mkdir -p /tmp/transformers_cache

# Set HuggingFace cache directories
ENV TRANSFORMERS_CACHE="/tmp/transformers_cache"
ENV HF_HOME="/tmp/transformers_cache"
ENV HF_HUB_CACHE="/tmp/transformers_cache"

# Expose port
EXPOSE 9000

# Start the FastAPI server
CMD ["python", "app.py"] 