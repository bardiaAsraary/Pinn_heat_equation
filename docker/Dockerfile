FROM python:3.11

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy all code
COPY . .

# Default command: run training
CMD ["python","src/train.py"]