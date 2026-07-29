# Use an official lightweight Python image with PyTorch compatibility
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer outputs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set container working directory
WORKDIR /app

# Install system dependencies needed for OpenCV / PyTorch image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for efficient Docker caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files into the container
COPY . .

# Grant execution permissions to entrypoint script
RUN chmod +x docker-entrypoint.sh

# Expose Streamlit and FastAPI ports
EXPOSE 8501
EXPOSE 8000

# Set container startup command
ENTRYPOINT ["./docker-entrypoint.sh"]