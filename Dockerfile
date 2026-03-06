# Base image: lightweight Python 3.11
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Upgrade pip first
RUN python -m pip install --upgrade pip

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy all project files
COPY . .

# Default command: run training script
CMD ["python", "src/train.py"]