FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

LABEL authors="YOUR_NAME"

# Set working directory
WORKDIR /app

# Update system packages
# RUN apt-get update -y && apt-get install -y \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender-dev \
#     libgomp1 \
#     && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
# RUN pip3 install --upgrade pip && \
#     pip3 install -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY tools/ tools/
COPY checkpoints/ checkpoints/
COPY main.py .
COPY model.py .

# Create writable directories
RUN mkdir -p /myhome && chmod 777 /myhome
ENV HOME=/myhome

# Ensure all files in /app are accessible
RUN chmod -R 777 /app

# Default command with required arguments
CMD ["python", "main.py", "-i", "/input", "-o", "/output", "-t", "seg", "-d", "gpu"]