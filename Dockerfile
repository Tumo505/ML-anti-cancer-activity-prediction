# Render.com Dockerfile for Anti-Cancer Drug Prediction App
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY pipeline.py .
COPY saved_model/ ./saved_model/
COPY sample*.csv ./

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
