FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and Git (for SAM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU PyTorch first (keeps image lightweight and avoids huge CUDA blobs)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code, static web assets, and checkpoints
COPY . .

# Expose backend port
EXPOSE 8000

# Start Uvicorn server on all interfaces
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
